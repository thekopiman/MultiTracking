import numpy as np
from .basemovement import BaseMovement
from ..objects import *


class LinearConstantVelocity(BaseMovement):
    def __init__(
        self, velocity: np.float32 = 10, direction=np.array([1, 1, 1], dtype=np.float32)
    ):
        self.velocity = velocity
        self.direction = self.direction_normalisation(direction)

    def update_velocity(self, velocity: np.float32):
        self.velocity = velocity

    def direction_normalisation(self, direction):
        norm = np.linalg.norm(direction)

        if norm != 0:
            v = direction / norm
        else:
            v = np.zeros_like(direction)

        return v

    def update_location(self, object: BaseObject, time: np.float32):
        u = object.current_location()
        v = u + self.direction * time
        object.update_location(v)

    def additive_vector(self, time):
        return self.direction * time
