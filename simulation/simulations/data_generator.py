import multiprocessing

import numpy as np
from numpy.random import SeedSequence, default_rng
import torch
from torch import Tensor

from TransformerMOT.util.misc import NestedTensor
from MOTSimulationV1 import MOTSimulationV1


class DataGenerator:
    def __init__(self, simulation_generator=MOTSimulationV1, batch=5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pool = multiprocessing.Pool()

        # Put this in Params in the future
        np.random.seed(0)

    def get_measurements(self):
        pass


def pad_to_batch_max(training_data, max_len):
    batch_size = len(training_data)
    d_meas = training_data[0].shape[1]
    training_data_padded = np.zeros((batch_size, max_len, d_meas))
    mask = np.ones((batch_size, max_len))
    for i, ex in enumerate(training_data):
        training_data_padded[i, : len(ex), :] = ex
        mask[i, : len(ex)] = 0

    return training_data_padded, mask


def get_single_training_example(data_generator):
    """Generates a single training example

    Returns:
        training_data   : A single training example
        true_data       : Ground truth for example
    """
    pass
