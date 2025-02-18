# import multiprocessing

import numpy as np
from numpy.random import SeedSequence, default_rng
import torch
from torch import Tensor

from TransformerMOT.util.misc import NestedTensor
from .MOTSimulationV1 import MOTSimulationV1


def attach_time(data: np.ndarray, interval):
    M, N, t, k = data.shape  # Extract current shape
    new_row = np.arange(t) * interval  # Shape (t,)

    # Reshape to (M, N, t, 1) for broadcasting
    new_row = new_row.reshape(1, 1, t, 1)

    # Repeat across M and N to match the shape
    new_row = np.tile(new_row, (M, N, 1, 1))

    # Concatenate along the last axis (k)
    return np.concatenate([data, new_row], axis=-1)


class DataGenerator:
    def __init__(
        self,
        params,
    ):
        self.params = params
        self.device = params.training.device
        # self.pool = multiprocessing.Pool()
        self.truncation = params.data_generation.truncation
        self.batch = int(params.training.batch_size)
        self.interval = params.data_generation.interval
        self.p = params.data_generation.p

        # Put this in Params in the future
        np.random.seed(params.training.seed)

        if params.data_generation.simulation_generator == "MOTSimulationV1":
            self.datagen = MOTSimulationV1(
                dimension=np.array(params.data_generation.dimension),
                sensor_radius=np.array(params.data_generation.sensor_radius),
                target_radius=np.array(params.data_generation.target_radius),
                ThreeD=params.data_generation.ThreeD,
                interval=self.interval,
            )

    def get_measurements(self, raw_data: tuple):
        """_summary_

        Args:
            raw_data (tuple): Containing =
            (
            truncated_sensors_timestamps,  # (M, t, 2/3)
            truncated_targets_timestamps,  # (N, t, 2/3)
            truncated_sensors_velocities,  # (M, t, 2/3)
            truncated_targets_velocities,  # (N, t, 2/3)
            truncated_angles_array,  # (M, N, t, 1/2)
            )

        Returns:
            training_nested_tensor (list): Each element consist of bearing-only
             measurements from the sensor together with the timestamp
            labels (list): Each element contains the expected coordinates and velocities
            of the targets
            unique_measuurement_ids (list): Target id corresponding to each bearing
            reading.
        """
        (
            truncated_sensors_timestamps,  # (M, t, 2/3)
            truncated_targets_timestamps,  # (N, t, 2/3)
            truncated_sensors_velocities,  # (M, t, 2/3) Redundant
            truncated_targets_velocities,  # (N, t, 2/3)
            truncated_angles_array,  # (M, N, t, 1/2)
        ) = raw_data
        truncated_angles_array = attach_time(truncated_angles_array, self.interval)

        batch_arr = [
            1 / 3 + 2 * (i) / (self.batch * 3) for i in range(1, self.batch + 1)
        ]
        batch_arr = [int(i * truncated_angles_array.shape[2]) for i in batch_arr]

        training_nested_tensor = []
        labels = []
        unique_measuurement_ids = []

        for end in batch_arr:
            split_sensors_timestamps = truncated_sensors_timestamps[:, :end, :]
            split_targets_timestamps = truncated_targets_timestamps[:, :end, :]
            # split_sensors_velocities = truncated_sensors_velocities[:end] # This is redundant
            split_targets_velocities = truncated_targets_velocities[:, :end, :]
            split_angles_array = truncated_angles_array[:, :end, :]

            t1, l1, u_id = self._step(
                split_sensors_timestamps,
                split_targets_timestamps,
                split_targets_velocities,
                split_angles_array,
            )

            training_nested_tensor.append(t1)
            labels.append(l1)
            unique_measuurement_ids.append(u_id)

        return training_nested_tensor, labels, unique_measuurement_ids

    def _step(
        self,
        split_sensors_timestamps,  # (M, t, 2/3)
        split_targets_timestamps,  # (N, t, 2/3)
        split_targets_velocities,  # (N, t, 2/3)
        split_angles_array,  # (M, N, t, 2/3) Time have been attached at the end
        #  of [:,:,:,-1]
    ):
        """Shuffling the mini dataset at each t"""
        total_duration = split_sensors_timestamps.shape[1]
        total_sensors = split_sensors_timestamps.shape[0]
        total_targets = split_targets_timestamps.shape[0]

        final_measurement = np.array([])
        final_unique_ids = np.array([], dtype="int64")

        for time in range(total_duration):
            measurement_array = []
            unique_target_ids = []
            for s in range(total_sensors):
                for t in range(total_targets):
                    # Target is not selected
                    if not self._bool_select_bearing():
                        break

                    bearing_and_time = split_angles_array[s, t, time, :]
                    sensor_coordinate = split_sensors_timestamps[s, time, :]
                    res = np.concatenate([sensor_coordinate, bearing_and_time])
                    measurement_array.append(res)
                    unique_target_ids.append(t)

            random_idx = np.random.permutation(len(measurement_array))
            measurement_array = np.array(measurement_array)[random_idx]
            unique_target_ids = np.array(unique_target_ids)[random_idx]

            # There might be situations where there are no bearing measurements in certain t

            if measurement_array.size > 0:
                final_measurement = (
                    np.vstack([final_measurement, measurement_array])
                    if final_measurement.size > 0
                    else measurement_array
                )

                final_unique_ids = np.hstack([final_unique_ids, unique_target_ids])

        label = np.concatenate(
            [
                split_targets_timestamps[:, -1, :],
                split_targets_velocities[:, -1, :],
            ],
            axis=-1,
        )

        return final_measurement, label, final_unique_ids

    def get_batch(self):
        self.raw_data = get_single_training_example(
            self.params, self.datagen, self.truncation
        )

        training_data, labels, unique_measurement_ids = self.get_measurements(
            self.raw_data
        )
        labels = [Tensor(l).to(torch.device(self.device)) for l in labels]
        unique_measurement_ids = [list(u) for u in unique_measurement_ids]

        # Pad training data
        max_len = max(list(map(len, training_data)))
        training_data, mask = pad_to_batch_max(training_data, max_len)

        # Pad unique ids
        for i in range(len(unique_measurement_ids)):
            unique_id = unique_measurement_ids[i]
            n_items_to_add = max_len - len(unique_id)
            unique_measurement_ids[i] = np.concatenate(
                [unique_id, [-2] * n_items_to_add]
            )[None, :]
        unique_measurement_ids = np.concatenate(unique_measurement_ids)

        training_nested_tensor = NestedTensor(
            Tensor(training_data).to(torch.float32).to(torch.device(self.device)),
            Tensor(mask).bool().to(torch.device(self.device)),
        )
        unique_measurement_ids = Tensor(unique_measurement_ids).to(self.device)

        return training_nested_tensor, labels, unique_measurement_ids

    def _bool_select_bearing(self) -> bool:
        x = np.random.uniform(0, 1)
        return x <= self.p


def pad_to_batch_max(training_data, max_len):
    batch_size = len(training_data)
    d_meas = training_data[0].shape[1]
    training_data_padded = np.zeros((batch_size, max_len, d_meas))
    mask = np.ones((batch_size, max_len))
    for i, ex in enumerate(training_data):
        training_data_padded[i, : len(ex), :] = ex
        mask[i, : len(ex)] = 0

    return training_data_padded, mask


def get_single_training_example(params, data_generator, truncation):
    """Generates a single training example

    Returns:
        training_data   : A single training example
        true_data       : Ground truth for example
    """
    data_generator.reset()
    data_generator.generate_checkpoints(
        no_targets_checkpoints=np.random.poisson(
            params.data_generation.checkpoints.targets
        ),
        no_sensors_checkpoints=np.random.poisson(
            params.data_generation.checkpoints.sensors
        ),
    )
    data_generator.spawn_sensors(
        distribution=lambda: max(
            np.random.poisson(params.data_generation.no_of_objects.sensors_lambda),
            params.data_generation.no_of_objects.min_sensors,
        ),
        error=lambda: np.random.poisson(
            params.data_generation.no_of_objects.sensor_error
        ),
    )
    data_generator.spawn_targets(
        distribution=lambda: max(
            np.random.poisson(params.data_generation.no_of_objects.targets_lambda),
            params.data_generation.no_of_objects.min_targets,
        ),
    )
    data_generator.generate_paths(
        sensor_speed_distribution=lambda: np.random.normal(
            params.data_generation.speed.sensors[0],
            params.data_generation.speed.sensors[1],
        ),
        target_speed_distribution=lambda: np.random.normal(
            params.data_generation.speed.targets[0],
            params.data_generation.speed.targets[1],
        ),
    )
    data_generator.run()

    truncated_sensors_timestamps = truncate_array(
        data_generator.sensors_timestamps, truncation
    )
    truncated_targets_timestamps = truncate_array(
        data_generator.targets_timestamps, truncation
    )
    truncated_sensors_velocities = truncate_array(
        data_generator.sensors_velocities, truncation
    )
    truncated_targets_velocities = truncate_array(
        data_generator.targets_velocities, truncation
    )
    truncated_angles = truncate_angles_array(data_generator.find_bearings(), truncation)

    return (
        truncated_sensors_timestamps,
        truncated_targets_timestamps,
        truncated_sensors_velocities,
        truncated_targets_velocities,
        truncated_angles,
    )


def truncate_array(arr, m):
    if arr.shape[1] > m:
        return arr[:, :m, :]
    return arr


def truncate_angles_array(arr, m):
    if arr.shape[2] > m:
        return arr[:, :, :m, :]
    return arr
