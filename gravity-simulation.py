from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class GravitySimulation:
    """Simulates the motion of n bodies under the influence of gravity using Newton's law of gravity"""

    def __init__(
        self,
        n_bodies: int = 2,
        masses: Iterable = [],
        gravity_strength: float = 1,
    ) -> None:
        """Initializes the class instance

        arguments:
            n_boides: number of bodies in the simulation. Defaults to 2.
            masses: relative masses of the bodies, defaults to equal masses.
            gravity_strength: strength of newtons force of gravity (analogous to big G).
                                Defaults to 1.
        """

        self.n_bodies = n_bodies
        self.masses = masses if masses else np.ones(n_bodies)
        self.gravity_strength = gravity_strength

        # Initialize position and velocity arrays
        self.pos = np.zeros((self.n_bodies, self.n_steps))
        self.vel = np.zeros((self.n_bodies, self.n_steps))


    def initialize(self, r0: Iterable, v0: Iterable, masses: Iterable) -> None:
        

    @staticmethod
    def newton_gravity(mass1: float, mass2: float, pos: np.ndarray) -> np.ndarray:
        """Calculates the force of Newtonian force of gravity between two bodies,
        from m1 unto m2. The inverse force from m2 unto m1 is the same magnitude, but opposite direction.

        arguments:
            m1: mass of body 1
            m2: mass of body 2
            r: vector from body 1 to body 2

        returns:
            np.ndarray: force vector from m1 unto m2
        """
        return -gravity_strength * mass1 * mass2 * pos / np.linalg.norm(pos) ** 3


if __name__ == "__main__":
    # Parameters
    time = 1000  # simulation time [s]
    time_step = 0.2  # time step [s]
    masses = [1, 1]  # body relative masses
    gravity_strength = 1  # strength of newtons force of gravity (analogous to big G)

    # # Set initial conditions for every body
    r0 = [[], []]
    r[0, 0] = -10
    r[1, 0] = 10
    v[0, 0] = 1
    v[1, 0] = -1

    sim = GravitySimulation(n_bodies, time, time_step)
    print(sim.masses)
    sim.initialize(r0, v0)
