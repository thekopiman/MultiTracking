import numpy as np
from simulation.objects import *


def find_theta_phi(s: Sensor, t: Target):
    dx, dy, dz = s.current_location() - t.current_location()
    return np.arctan2(dy, dx), np.arctan(dz / np.sqrt(dx**2 + dy**2))


class Simulation:
    def __init__(
        self,
        boundary: np.array = np.array([[-10, 10], [-10, 10], [-10, 10]], interval=0.01),
    ):
        self.boundary = boundary
        self.sensors = []
        self.targets = []
        self.calculate_angles = None
        self.interval = 0.01

    def add_sensors(self, sensor: Sensor):
        self.sensors.append(sensor)

    @PendingDeprecationWarning
    def print_sensors(self):
        for idx, i in enumerate(self.sensors):
            print(f"{idx} : {i}")

    def remove_sensor(self, index):
        assert len(self.sensors) > 0
        self.sensors.remove(self.sensors[index])

    def add_targets(self, target: Target):
        self.targets.append(target)

    @PendingDeprecationWarning
    def print_targets(self):
        for idx, i in enumerate(self.targets):
            print(f"{idx} : {i}")

    def remove_target(self, index):
        assert len(self.targets) > 0
        self.targets.remove(self.targets[index])

    @PendingDeprecationWarning
    def __repr__(self):
        print("--- Sensors ---")
        self.print_sensors()
        print("--- Targets ---")
        self.print_targets()
        return ""

    @DeprecationWarning
    def find_bearings(self):
        assert len(self.sensors) > 0
        self.calculate_angles = np.zeros((len(self.sensors), len(self.targets), 2))

        for idx_s, s in enumerate(self.sensors):
            for idx_t, t in enumerate(self.targets):
                self.calculate_angles[idx_s, idx_t, :] = find_theta_phi(s, t)

        return self.calculate_angles
