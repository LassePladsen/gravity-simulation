from typing import Iterable, TypeAlias
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation

Pathlike: TypeAlias = str | Path


def newton_gravity(
    pos_vector: np.ndarray, masses: Iterable[float], gravity_strength: float = 1
) -> np.ndarray:
    """Calculates the force of Newtonian force of gravity between two bodies, on body 1 from body 2.
    The inverse force from body 1 unto body 2 is the same magnitude, but opposite direction.

    arguments:
        pos_vector: positional vector from body 1 to body 2
        masses: relative masses of two bodies [mass1, mass2]
        gravity_strength: strength of newtons force of gravity (analogous to big G). Defaults to 1.

    returns:
        np.ndarray: force vector on body 1 from body 2
    """
    return (
        gravity_strength
        * masses[0]
        * masses[1]
        * pos_vector
        / np.linalg.norm(pos_vector) ** 3
    )


class GravitySim2D:
    """2D gravity simulation of bodies using Newton's law of gravity, with collision."""

    def __init__(
        self,
        time: float,
        time_step: float,
        n_bodies: int = 2,
        masses: Iterable = [],
        gravity_strength: float = 1,
    ) -> None:
        """Initializes the class instance

        arguments:
            time: simulation time [s]
            time_step: time step [s]
            n_bodies: number of bodies in the simulation. Defaults to 2.
            masses: relative masses of the bodies, defaults to equal masses.
            gravity_strength: strength of newtons force of gravity (analogous to big G).
                                Defaults to 1.
        """

        if n_bodies < 2:
            raise ValueError("n_bodies must be greater than one.")

        # Set instance variables
        self.time = time
        self.time_step = time_step
        self.n_steps = int(time / time_step)
        self.n_bodies = n_bodies
        self.masses = masses if masses else np.ones(n_bodies)
        self.gravity_strength = gravity_strength

        # Initialize position and velocity arrays
        self.t = np.arange(0, time, time_step)  # time array
        self.reset()

        self._break = False  # flag to break the simulation

    def reset(self) -> None:
        """Resets the position and velocity arrays to zero. Shape: [2, n_bodies, n_steps]

        arguments:
            None

        returns:
            None
        """
        self.pos = np.zeros((self.n_steps, self.n_bodies, 2))
        self.vel = np.zeros((self.n_steps, self.n_bodies, 2))

    def simulate(
        self, pos_init: Iterable, vel_init: Iterable
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulates the motion of the bodies under the influence of gravity

        arguments:
            pos_init: initial position of all bodies with shape (self.n_bodies, 2)
            vel_init: initial velocity of all bodies with shape (self.n_bodies, 2)

        returns:
            np.ndarray: the time array, and the position array of the bodies at
                        each time step with shape (self.n_steps, self.n_bodies, 2)and the
        """
        # Cast Iterables to np.ndarrays
        if not isinstance(pos_init, np.ndarray):
            pos_init = np.array(pos_init)
        if not isinstance(vel_init, np.ndarray):
            vel_init = np.array(vel_init)

        # Check if array shapes are correct
        if pos_init.shape != (self.n_bodies, 2) or pos_init.shape != (self.n_bodies, 2):
            raise ValueError(
                "Initial position and velocity arrays must have shape (n_bodies, 2)"
            )

        # Set initial conditions
        self.pos[0] = pos_init
        self.vel[0] = vel_init

        self._break = False
        # Simulate the motion of the bodies
        for step in range(self.n_steps - 1):
            if self._break:
                break
            self._step(step)

        return self.t, self.pos

    def _step(self, step: int = 0) -> None:
        """Performs a single time step in the simulation using the Euler-Cromer method,
        newtons law of gravity, and newtons third law

        arguments:
            step: step number

        return:
            None
        """
        for i in range(self.n_bodies - 1):
            for j in range(i + 1, self.n_bodies):
                dist_vector = self.pos[step, j] - self.pos[step, i]

                # Skip if they are at the same position
                pos_min = 0.1
                if np.linalg.norm(dist_vector) < pos_min:
                    continue

                force = newton_gravity(dist_vector, self.masses, self.gravity_strength)

                # Body i
                self.vel[step + 1, i] = (
                    self.vel[step, i] + force / self.masses[i] * self.time_step
                )
                self.pos[step + 1, i] = (
                    self.pos[step, i] + self.vel[step + 1, i] * self.time_step
                )

                # Body j
                self.vel[step + 1, j] = (
                    self.vel[step, j] - force / self.masses[j] * self.time_step
                )
                self.pos[step + 1, j] = (
                    self.pos[step, j] + self.vel[step + 1, j] * self.time_step
                )

                # If collision on this step: use inelastic momentum conservation (they stick together)
                if (
                    np.linalg.norm(self.pos[step + 1, i] - self.pos[step + 1, j])
                    < pos_min
                ):
                    v1 = self.vel[step + 1, i]
                    v2 = self.vel[step + 1, j]

                    # New shared velocity
                    denum = m1 * v1 + m2 * v2
                    if all(denum) == 0:  # momentum cancels out: they must stop
                        new_vel = np.zeros_like(v1)
                        self.pos[step + 1 :, i] = self.pos[step + 1, i]
                        self.pos[step + 1 :, j] = self.pos[step + 1, j]
                    else:
                        new_vel = (m1 + m2) / (denum)
                    self.vel[step + 1, i] = new_vel
                    self.vel[step + 1, j] = new_vel

                # DEBUG: Stop early if they are extremely far away from each other
                pos_max = 1000
                if (
                    np.linalg.norm(self.pos[step + 1, i] - self.pos[step + 1, j])
                    > pos_max
                ):
                    # Replace rest of array with these last values
                    self.pos[step + 1 :] = self.pos[step + 1]
                    self._break = True
                    return

    def plot1d(
        self,
        axis: int | str = "both",
        filename: Pathlike = "",
        figsize: tuple[int, int] = (7, 5),
    ) -> None:
        """Plots the 2D motion of the bodies

        arguments:
            axis: axis to plot (0=x, 1=y, "both"=x and y). Defaults to both.
            filename: filename to save the plot to. If none, shows the plot instead.

        returns:
            None
        """
        if axis not in [0, 1, "both"]:
            raise ValueError(
                "axis must be either 0, 1, or 'both' (respectively x-axis or y-axis, or both)"
            )

        if not isinstance(filename, (str, Path)):
            raise ValueError("filename must be a string or a pathlib.Path object")

        # Plotting
        if axis == "both":
            fig, axs = plt.subplots(2, 1, figsize=figsize)
            for i in range(n_bodies):
                axs[0].plot(self.t, self.pos[:, i, 0], label=f"body {i}")
                axs[1].plot(self.t, self.pos[:, i, 1], label=f"body {i}")
            axs[0].set_ylabel("x")
            axs[1].set_ylabel("y")
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            for i in range(n_bodies):
                ax.plot(self.t, self.pos[:, i, axis], label=f"body {i}")
            ylabel = "x" if axis == 0 else "y"
            plt.ylabel(ylabel)

        # Config
        fig.supxlabel("Time [s]")
        plt.legend()

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def _set_axis_limits(self, ax: plt.Axes, frame: int, scale: float = 5) -> None:
        """Sets the axis limits to fit all bodies in the plot,
        if they are significantly outside/inside the current limits

        arguments:
            ax: the matplotlib axis object
            frame: the current frame number
            scale: scale factor to add to the limits. Defaults to 5.

        returns:
            None
        """

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        # Multiply each limits by the scale factor if its outside the limits (keep equal axis scale)
        if (
            any(self.pos[frame, :, 0] < x0)
            or any(self.pos[frame, :, 0] > x1)
            or any(self.pos[frame, :, 1] < y0)
            or any(self.pos[frame, :, 1] > y1)
        ):
            x0 -= abs(x0 * scale)
            ax.set_xlim(x0, -x0)
            y0 -= abs(y0 * scale)
            ax.set_ylim(y0, -y0)

    def animate(
        self,
        ms: float = None,
        filename: Pathlike = "",
        figsize: tuple[int, int] = (7, 5),
    ) -> None:
        """Animates the 2D motion of the bodies

        arguments:
            ms: milliseconds delay between frames. Defaults to self.time_step.
            filename: filename to save the animation to. If none, shows the animation instead.
            figsize: figure size. Defaults to (7, 5).

        returns:
            None
        """
        if ms is None:
            ms = self.time_step

        fig, ax = plt.subplots(figsize=figsize)

        # Plot initial positions
        scatters = [ax.scatter(*self.pos[0, i]) for i in range(self.n_bodies)]

        # Initial axis limits
        self._set_axis_limits(ax, 0)

        def update(frame: int):
            for i, scatter in enumerate(scatters):
                scatter.set_offsets(self.pos[frame, i])

            # Update title with time
            fig.suptitle(f"Time: {self.t[frame]:.2f} s")

            # Update axis limits to fit all bodies
            self._set_axis_limits(ax, frame)

        # Create animation
        anim = FuncAnimation(fig, update, frames=len(self.t), interval=ms)

        # Config
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")

        # Save or show animation
        if filename:
            # anim.save(filename)
            video = anim.to_html5_video()
            video.save(filename)
        else:
            plt.show()


if __name__ == "__main__":
    # Parameters
    n_bodies = 2  # number of bodies in the simulation
    time = 100  # simulation time [s]
    time_step = 0.1  # time step [s]
    masses = [1, 10]  # relative body masses
    gravity_strength = 2  # strength of newtons force of gravity (analogous to big G)

    # # Set initial conditions for every body
    r0 = [[0, 0], [0, 10]]  # initial positions of the bodies (x0, y0), (x1, y1) etc.
    v0 = [[1, 1], [0, -1]]  # initial velocities

    sim = GravitySim2D(
        time=time,
        time_step=time_step,
        n_bodies=n_bodies,
        masses=masses,
        gravity_strength=gravity_strength,
    )
    sim.simulate(r0, v0)
    # sim.plot1d()
    # sim.animate(filename="test.gif")
    sim.animate()
