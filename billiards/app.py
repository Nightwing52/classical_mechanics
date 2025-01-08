import numpy as np

class BilliardsSimulation:
    def __init__(self, positions, velocities, time_step=0.01):
        """
        positions: (n, 2) array where each row is [x, y] for a ball
        velocities: (n, 2) array where each row is [vx, vy] for a ball
        time_step: float, time step for the simulation
        """
        self.positions = positions
        self.velocities = velocities
        self.time_step = time_step

    def update(self):
        # next step
        next_step = self.positions+self.velocities*self.time_step

        # Reflect at walls (x-direction)
        mask_x = (next_step[:, 0] <= 0) | (next_step[:, 0] >= 1)
        print(mask_x)
        self.velocities[mask_x, 0] *= -1

        # Reflect at walls (y-direction)
        mask_y = (next_step[:, 1] <= 0) | (next_step[:, 1] >= 1)
        self.velocities[mask_y, 1] *= -1
        
        # Update positions
        self.positions += self.velocities * self.time_step

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm

class BilliardsVisualizer:
    def __init__(self, simulation):
        self.simulation = simulation

    def animate(self, frames=100, fileName="output.gif"):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Generate a unique color for each ball based on a colormap
        num_balls = len(self.simulation.positions)
        colors = cm.viridis(np.linspace(0, 1, num_balls))  # Use 'viridis' colormap

        scatter = ax.scatter([], [])

        def init():
            scatter.set_offsets(self.simulation.positions)
            scatter.set_color(colors)
            return scatter,

        def update(frame):
            self.simulation.update()
            scatter.set_offsets(self.simulation.positions)
            return scatter,

        ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
        save_path = "/app/output/"+fileName
        if save_path.endswith('.gif'):
                writer = PillowWriter(fps=30)
                ani.save(save_path, writer=writer)
        else:
            raise ValueError("Unsupported file format. Use '.gif' or '.mp4'.")

# Initialize positions and velocities as NumPy arrays
positions = np.array([
    [0.2, 0.3],
    [0.7, 0.8]
])

velocities = np.array([
    [1.0, 2.0],
    [-2.0, -1.0]
])

# Create simulation and visualizer
sim = BilliardsSimulation(positions, velocities)
viz = BilliardsVisualizer(sim)

# Animate the simulation
viz.animate(frames=200)
