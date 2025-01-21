import numpy as np
from ..objects import *


class BaseMovement:
    def __init__(self):
        pass

    def update_location(self, object: BaseObject, time: np.float32):
        pass

    def additive_vector(self, time):
        return np.array([0, 0, 0])
