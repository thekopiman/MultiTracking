import numpy as np
from .baseobject import BaseObject


class Sensor(BaseObject):
    def __init__(self, initial_location: np.array = np.zeros((3))):
        super().__init__(initial_location)
