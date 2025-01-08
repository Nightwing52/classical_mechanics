from app import BilliardsSimulation
import numpy as np

def test_stationary():
    # Initialize positions and velocities as NumPy arrays
    positions = np.array([
        [0.2, 0.3]
    ])

    velocities = np.array([
        [0.0, 0.0]
    ])

    sim = BilliardsSimulation(positions, velocities)
    sim.update()
    assert(np.allclose(sim.positions, positions)) # ball should not move