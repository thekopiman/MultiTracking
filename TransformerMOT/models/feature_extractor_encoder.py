import torch
import torch.nn as nn


@DeprecationWarning
class RangeParameterizationLayer(nn.Module):
    def __init__(self, num_d=10):
        super().__init__()
        self.num_d = num_d
        self.d_embeddings = nn.Embedding(num_d, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.d_embeddings.weight)

    def forward(self, x):
        B, C, T = x.shape

        assert C == 5 or C == 3  # 2D or 3D only including angles

        embed = self.d_embeddings.weight
        embed = embed.view(1, 1, self.num_d, 1)

        # 3D
        if C == 5:
            SensorCoordinates = x[:, :3, :]  # (B, 3, T)

            azimuth = x[:, 3, :]
            elevation = x[:, 4, :]

            x_polar = torch.sin(elevation) * torch.cos(azimuth)
            y_polar = torch.sin(elevation) * torch.sin(azimuth)
            z_polar = torch.cos(elevation)

            polar_cartesian = torch.stack([x_polar, y_polar, z_polar], dim=1)

            polar_cartesian = polar_cartesian.unsqueeze(-2)
            transformed_polar = polar_cartesian * embed

            SensorCoordinates = SensorCoordinates.unsqueeze(-2).expand(
                -1, -1, self.num_d, -1
            )

            output = SensorCoordinates + transformed_polar

            return output

        # 2D
        elif C == 3:
            SensorCoordinates = x[:, :2, :]  # (B, 2, T)

            azimuth = x[:, 2, :]

            x_polar = torch.cos(azimuth)
            y_polar = torch.sin(azimuth)

            polar_cartesian = torch.stack([x_polar, y_polar], dim=1)

            polar_cartesian = polar_cartesian.unsqueeze(-2)
            transformed_polar = polar_cartesian * embed

            SensorCoordinates = SensorCoordinates.unsqueeze(-2).expand(
                -1, -1, self.num_d, -1
            )

            output = SensorCoordinates + transformed_polar

            return output


class SelfAttentionFeatureExtractor(nn.Module):
    """Self-attention to reduce feature dimension (d → d_model)"""

    def __init__(self, d_input, d_model):
        super().__init__()
        self.qkv_proj = nn.Linear(
            d_input, d_model * 3, dtype=torch.float32
        )  # Project to Q, K, V
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Linear(
            d_model, d_model, dtype=torch.float32
        )  # Final projection

    def forward(self, x, mask):
        # x: (B, 3, d, t)
        B, C, d, t = x.shape
        x = x.permute(0, 1, 3, 2)  # Reshape to (B, cartesian_dim, t, d) for attention

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (B, cartesian_dim, t, 3*d_model)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Split into Q, K, V

        # Compute attention weights
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (
            q.shape[-1] ** 0.5
        )  # (B, cartesian_dim, t, t)

        # Apply Mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = self.softmax(attn_scores)  # (B, cartesian_dim, t, t)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)  # (B, cartesian_dim, t, d_model)

        # Final projection
        output = self.output_proj(attn_output)  # (B, cartesian_dim, t, d_model)
        return output


@DeprecationWarning
class FeatureExtractorEncoder(nn.Module):
    def __init__(
        self,
        d_input,
        d_model,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout=0.1,
        d_detection=5,
    ):
        super().__init__()

        # Self-Attention Feature Extraction
        self.feature_extractor = SelfAttentionFeatureExtractor(d_input, d_model)

        cartesian_dim = d_detection // 2 + 1
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model * cartesian_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ),
            num_layers=num_layers,
        )

        # Project back to Cartesian coordinates (d_model → 3)
        self.output_projection = nn.Linear(d_model * cartesian_dim, cartesian_dim)

    def forward(self, x, src_mask=None):
        # x: (B, cartesian_dim, d, t)
        B, C, d, t = x.shape

        # Feature Extraction (Self-Attention reduces d → d_model)
        x = self.feature_extractor(x, mask=src_mask)  # (B, cartesian_dim, t, d_model)

        # Reshape to merge Cartesian channels into feature space
        x = x.reshape(B, t, -1)  # (B, t, cartesian_dim * d_model)

        # Apply Transformer
        x = self.transformer(
            x, src_key_padding_mask=src_mask
        )  # (B, t, cartesian_dim * d_model)

        # Project back to Cartesian (B, t, cartesian_dim)
        x = self.output_projection(x)  # (B, t, cartesian_dim)

        # Reshape back to (B, cartesian_dim, t)
        return x.permute(0, 2, 1)


if __name__ == "__main__":
    # # Example Usage
    B, C, t = (
        8,
        5,
        50,
    )  # Batch size 8, 3D coordinates, sequence length 50
    x = torch.randn(B, C, t)
    d_num = 64

    # model = FeatureExtractorEncoder(
    #     d_input=64, d_model=32, num_heads=8, num_layers=4, dim_feedforward=128
    # )
    # output = model(x)  # (B, 3, t)
    # print(output.shape)  # Should be (8, 3, 50)

    RF_Layer = RangeParameterizationLayer(d_num)
    model1 = FeatureExtractorEncoder(
        d_input=d_num, d_model=32, num_heads=8, num_layers=4, dim_feedforward=128
    )

    x = RF_Layer(x)
    print(x.shape)
    # x = SA_FE_layer(x)
    # print(x.shape)
    output = model1(x)
    print(output.shape)
