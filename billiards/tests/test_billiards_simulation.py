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

def test_ball_movement():
    positions = np.array([[0.5, 0.5]])
    velocities = np.array([[0.1, 0.2]])
    sim = BilliardsSimulation(positions, velocities, time_step=1.0)

    sim.update()

    expected_positions = np.array([[0.6, 0.7]])
    assert np.allclose(sim.positions, expected_positions)

def test_wall_reflection():
    positions = np.array([[0.99, 0.5]])
    velocities = np.array([[0.2, 0.1]])
    sim = BilliardsSimulation(positions, velocities, time_step=1.0)

    sim.update()

    # Ball should reflect in the x direction
    expected_velocities = np.array([[-0.2, 0.1]])
    assert np.allclose(sim.velocities, expected_velocities)

def test_boundary_behavior():
    positions = np.array([[0.99, 0.99]])
    velocities = np.array([[0.1, 0.1]])
    sim = BilliardsSimulation(positions, velocities, time_step=1.0)

    sim.update()

    # Ball should reflect and stay within the unit square
    assert np.all(sim.positions >= 0) and np.all(sim.positions <= 1)

def test_multiple_balls():
    positions = np.array([[0.5, 0.5], [0.1, 0.9]])
    velocities = np.array([[0.1, -0.1], [-0.05, 0.05]])
    sim = BilliardsSimulation(positions, velocities, time_step=1.0)

    sim.update()

    expected_positions = np.array([[0.6, 0.4], [0.05, 0.95]])
    assert np.allclose(sim.positions, expected_positions)

