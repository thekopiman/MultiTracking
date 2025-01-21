import numpy as np


class BaseObject:
    def __init__(self, initial_location: np.array = np.zeros((3))):
        self.location = initial_location

        # TBC
        self.previous_movement = None

    def current_location(self):
        return self.location

    def __repr__(self) -> np.array:
        return f"{self.current_location()}"

    def update_location(self, new_location: np.array):
        self.location = new_location


if __name__ == "__main__":
    obj = BaseObject()
