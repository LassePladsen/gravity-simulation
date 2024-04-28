from typing import Iterable, TypeAlias
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
            masses: relative masses of the bodies. Defaults to equal masses.
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
        if not masses:
            self.masses = np.ones(n_bodies)
        else:
            self.masses = np.asarray(masses) / sum(masses)  # normalize masses
        self.gravity_strength = gravity_strength

        # Initialize position and velocity arrays
        self.t = np.arange(0, time, time_step)  # time array
        self.reset()

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

        # Simulate the motion of the bodies
        for step in range(self.n_steps - 1):
            self._step(step)

        return self.t, self.pos

    def _normal_step_update(self, step: int, i: int, force_vector: np.ndarray) -> None:
        """Updates the position and velocity for a given body i, for a normal time step
            without collision

        arguments:
            step: step number
            i: index for body to update
            force_vector: force vector on body i from all other bodies

        returns:
            None
        """
        # Update velocity
        self.vel[step + 1, i] = (
            self.vel[step, i] + force_vector / self.masses[i] * self.time_step
        )

        # Update position
        self.pos[step + 1, i] = (
            self.pos[step, i] + self.vel[step + 1, i] * self.time_step
        )

    def _step_collision_update(
        self, step: int, i: int, j: int, dist_vector: np.ndarray
    ):
        """Updates the position and velocity arrays for a time step with collision

        arguments:
            step: step number
            i: index for body 1
            j: index for body 2
            dist_vector: vector from body i to body j

        returns:
            None
        """

        veli = self.vel[step, i]
        velj = self.vel[step, j]

        # Case 1: they are already moving together: skip collision calculations
        if self.n_bodies == 2 and all(veli == velj):
            return

        # Case 2: They now stick together with the same velocity
        massi = self.masses[i]
        massj = self.masses[j]
        denominator = massi * veli + massj * velj

        # Momentum cancels out: they must stop
        if all(denominator == 0):
            newpos = self.pos[step, i] + dist_vector / 2  # they meet halfway
            self.pos[step + 1, i] = newpos
            self.pos[step + 1, j] = newpos

            # New velocities are zero
            self.vel[step + 1] = np.zeros((self.n_bodies, 2))
            return

        # Momentum does not cancel out; they still move together
        # Calculate this new shared velocity
        np.seterr(divide="ignore")  # ignore division by zero warning
        new_vel = (massi + massj) / (denominator)
        new_vel[new_vel == np.inf] = 0  # replace inf with zero

        # Update velocities from collision
        self.vel[step + 1, i] = new_vel
        self.vel[step + 1, j] = new_vel

        # Update positions from collision
        new_pos = self.pos[step, i] + new_vel * self.time_step
        self.pos[step + 1, i] = new_pos
        self.pos[step + 1, j] = new_pos

    def _step(self, step) -> None:
        """Performs a single time step in the simulation using the Euler-Cromer method,
        newtons law of gravity, and newtons third law

        arguments:
            step: step number

        return:
            None
        """

        for i in range(self.n_bodies - 1):
            forcei = 0
            forcej = 0
            for j in range(i + 1, self.n_bodies):
                posi = self.pos[step, i]
                posj = self.pos[step, j]
                dist_vector = posj - posi

                # Check for collision
                pos_tol = 1  # position tolerance for collision
                skip = False

                # Collision: use inelastic momentum conservation (they stick together)
                if np.linalg.norm(dist_vector) < pos_tol:
                    self._step_collision_update(step, i, j, dist_vector)
                    skip = True

                # No collision: accumulate force vector on body i from all other bodies
                if not skip:
                    change = newton_gravity(
                        dist_vector,
                        [self.masses[i], self.masses[j]],
                        self.gravity_strength,
                    )
                    forcei += change
                    forcej -= change

                # No collision; normal gravity update
                if not skip:
                    self._normal_step_update(step, i, forcei)
                    self._normal_step_update(step, j, forcej)

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
            for i in range(self.n_bodies):
                axs[0].plot(self.t, self.pos[:, i, 0], label=f"body {i}")
                axs[1].plot(self.t, self.pos[:, i, 1], label=f"body {i}")
            axs[0].set_ylabel("x")
            axs[1].set_ylabel("y")
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            for i in range(self.n_bodies):
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

    def _set_axis_limits(
        self, ax: plt.Axes, frame: int, grow: float = 5, shrink: float = 51 / 100
    ) -> None:
        """Sets the axis limits to fit all bodies in the plot,
        if they are significantly outside/inside the current limits

        arguments:
            ax: the matplotlib axis object
            frame: the current frame number
            grow: grow scale factor to multiply limits by.
            shrink: shrink scale factor. This is the ratio of the current square
                    to set the new limits from as [shrink * (x1-x0, y1-y0)]
                    Must be between 0 and 1.

        returns:
            None
        """
        if not 0 < shrink < 1:
            raise ValueError("shrink ratio must be between 0 and 1")

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        # Handle the cases where bodies exit the limits/the figure
        if (
            any(self.pos[frame, :, 0] < x0)
            or any(self.pos[frame, :, 0] > x1)
            or any(self.pos[frame, :, 1] < y0)
            or any(self.pos[frame, :, 1] > y1)
        ):
            x0 -= abs(x0 * grow)
            ax.set_xlim(x0, -x0)
            y0 -= abs(y0 * grow)
            ax.set_ylim(y0, -y0)

        # Handle the case where the limits are too big for the current body positions
        # This will be when all bodies are inside a certain fraction of total limit square
        dx = (1 - shrink) * (x1 - x0)
        dy = (1 - shrink) * (y1 - y0)
        x0 += dx
        x1 -= dx
        y0 += dy
        y1 -= dy
        if (
            all(self.pos[frame, :, 0] > x0)
            and all(self.pos[frame, :, 0] < x1)
            and all(self.pos[frame, :, 1] > y0)
            and all(self.pos[frame, :, 1] < y1)
        ):
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

    def animate(
        self,
        ms_interval: float = None,
        filename: Pathlike = "",
        figsize: tuple[int, int] = (7, 5),
        auto_axes: bool = True,
        **kwargs: dict,
    ) -> None:
        """Animates the 2D motion of the bodies

        arguments:
            ms_interval: milliseconds delay between frames. Defaults to self.time_step.
            filename: filename to save the animation to. If none, shows the animation instead.
            figsize: figure size. Defaults to (7, 5).
            auto_axes: flag to automatically adjust axis limits. Defaults to True.
            kwargs: additional keyword arguments to pass to FuncAnimation.save()

        returns:
            None
        """

        if ms_interval is None:
            ms_interval = self.time_step

        fig, ax = plt.subplots(figsize=figsize)

        # Plot initial positions
        scatters = [ax.scatter(*self.pos[0, i]) for i in range(self.n_bodies)]

        # Initial axis limits
        self._set_axis_limits(ax, 0) if auto_axes else None

        def update(frame: int):
            for i, scatter in enumerate(scatters):
                scatter.set_offsets(self.pos[frame, i])

            # Update title with time
            fig.suptitle(f"Time: {self.t[frame]:.2f} s")

            # Update axis limits to fit all bodies
            self._set_axis_limits(ax, frame) if auto_axes else None

        # Create animation
        anim = FuncAnimation(fig, update, frames=len(self.t), interval=ms_interval)

        # Config
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")

        # Save or show animation
        if filename:
            anim.save(filename, **kwargs)
        else:
            plt.show()


if __name__ == "__main__":
    ### EXAMPLE USAGE

    # Parameters
    time = 500  # simulation time [s]
    time_step = 1  # time step [s]
    gravity_strength = 1e4  # strength of newtons force of gravity (analogous to big G)

    masses = [  # relative body mass ratios
        1,
        1,
        1,
    ]

    r0 = [  # Initial positions
        [0, 0],
        [-150, 0],
        [100, 0],
    ]
    v0 = [  # Initial velocities
        [0, 5],
        [3, 4],
        [-1, 1],
    ]

    sim = GravitySim2D(
        time=time,
        time_step=time_step,
        n_bodies=len(masses),
        masses=masses,
        gravity_strength=gravity_strength,
    )
    sim.simulate(r0, v0)
    # sim.plot1d()
    # sim.animate(interval=0)
    sim.animate(filename="example.gif", fps=30, dpi=300)

    """Warning: saving the animation to a file will take a while for ffmpeg to 
    process if the simulation time is long.
    Inputting no filename will instead show the animation in a window, which is instant"""
