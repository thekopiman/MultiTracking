import torch
import torch.nn as nn


def find_closest_index(tensor_sequence, x):
    return torch.argmin(torch.abs(tensor_sequence - x), dim=-1)


class RPFlooding:
    def __init__(self, params):
        self.d_radius = params.arch.rp_flooding.d_radius
        self.start = params.data_generation.dimension[0][0]
        self.end = params.data_generation.dimension[0][1]
        self.device = params.training.device
        self.d_range = torch.linspace(self.start, self.end, self.d_radius).to(
            self.device
        )

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor,
        target_coordinates=None,
        unique_id=None,
    ):
        """_summary_

        Args:
            src (torch.Tensor): _description_
            mask (torch.Tensor): _description_
            target_coordinates (_type_, optional): _description_. Defaults to None.
            unique_id (_type_, optional): _description_. Defaults to None.

        Returns:
            src_new: (B, t*d, feature_dim + 1)
            optim_indices: (B, t * d, 1) (with -1 for noise/clutter and correct unique_id for targets)
        """
        # Input is (B, t, sensor xy + angle + timestamp)
        B, t, _ = src.shape

        self.d_range_expanded = self.d_range.view(1, 1, -1).expand(
            B, t, -1
        )  # (B, t, d)

        # Apply mask | mask = True will cause d = 0
        self.d_range_expanded = torch.where(
            mask.unsqueeze(-1),
            torch.zeros_like(self.d_range_expanded).to(mask.device),
            self.d_range_expanded.to(mask.device),
        )

        self.d_range_expanded = self.d_range_expanded.reshape(B, -1, 1)  # (B, t * d, 1)

        src_new = src.repeat_interleave(repeats=self.d_radius, dim=1)  # (B, t * d, _)
        src_new = torch.cat(
            (src_new, self.d_range_expanded), dim=-1
        )  # (B, t * d, _ + 1)

        # Swap timestamp and d
        src_new[:, :, [-1, -2]] = src_new[:, :, [-2, -1]]
        sensor_xy = src[:, :, :2]  # Have to change to 2/3 cartesian format next time

        optim_indices = None

        if target_coordinates is not None and unique_id is not None:
            optim_indices = (torch.zeros((B, t * self.d_radius, 1)) - 1).to(
                self.device
            )  # All clutter/noise first with shape (B, t * d, 1)

            distance = torch.abs(
                torch.linalg.norm(target_coordinates - sensor_xy, dim=-1, keepdim=True)
            )  # (B, t, 1)

            closest_indices = find_closest_index(self.d_range, distance)  # (B, t)
            closest_indices = (
                closest_indices.unsqueeze(-1)
                .repeat_interleave(self.d_radius, dim=-1)
                .to(self.device)
            )  # Convert to (B, t, d)

            # Expand unique_id to shape (B, t, 1) if needed
            unique_id = unique_id.view(B, t, 1)

            # Repeat interleave the unique_id so that each expanded (t*d) gets assigned properly
            unique_id_expanded = unique_id.repeat_interleave(
                self.d_radius, dim=1
            )  # (B, t * d, 1)

            # Generate a mask to set the correct indices in optim_indices
            idx_mask = torch.arange(self.d_radius).view(1, 1, -1)  # (1, 1, d)
            idx_mask = idx_mask.expand(B, t, -1).to(self.device)  # (B, t, d)

            # Create a mask where the closest index matches the d_range index
            selection_mask = (idx_mask == closest_indices).to(
                self.device
            )  # (B, t, d), True where index matches

            # Reshape mask to match (B, t*d, 1)
            selection_mask = selection_mask.reshape(B, t * self.d_radius, 1)

            # Assign unique_id where selection_mask is True
            optim_indices[selection_mask] = unique_id_expanded[selection_mask]
            optim_indices = optim_indices.squeeze(dim=-1)

        return src_new, optim_indices
