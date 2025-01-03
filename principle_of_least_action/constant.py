import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Number of interior points
TOTAL_POINTS = 100
n_points = TOTAL_POINTS-2  # Total points is n_points + 2 (boundary points)
delta_x = 1.0/(TOTAL_POINTS-1)

test_x = np.linspace(0, 1, 10)
test_y = np.linspace(0, 1, 10)**2
print("x: "+str(test_x))
print("y: "+str(test_y))
print("y': "+str(np.gradient(test_y, 1.0/9.0)))

# Define the objective function for optimization
def constant_potential(x_inner):
    x = np.concatenate(([0], x_inner, [1]))  # Add fixed boundary points
    x_dot = np.gradient(x, delta_x)  # Compute the gradient
    return np.sum(x_dot**2)  # Minimize the sum of squares of the gradient

def constant_gravity(x_inner):
    x = np.concatenate(([0], x_inner, [0]))  # Add fixed boundary points
    x_dot = np.gradient(x, delta_x)  # Compute the gradient
    return np.sum(x_dot**2)  # Minimize the sum of squares of the gradient

def harmonic_oscillator(x_inner):
    x = np.concatenate(([0], x_inner, [0]))  # Add fixed boundary points
    x_dot = np.gradient(x, delta_x)  # Compute the gradient
    return np.sum(x_dot**2)  # Minimize the sum of squares of the gradient

# Energy constraint: ensure x_dot^2 at each time step matches the initial value
def energy_constraint(x_inner):
    x = np.concatenate(([0], x_inner, [1]))
    x_dot = np.gradient(x, delta_x)
    return x_dot**2 - x_dot_initial**2

# Initial guess for interior points of x
x_inner_initial = np.linspace(0, 1, n_points + 2)[1:-1]

# Compute initial momentum and energy
x_initial = np.concatenate(([0], x_inner_initial, [1]))**2
t = np.linspace(0, 1, len(x_initial))
x_dot_initial = (1.0-0.0)/1.0
print(x_dot_initial)
initial_energy = x_dot_initial**2 # np.sum(x_dot_initial**2)

# Define constraints
constraints = [
    {"type": "eq", "fun": energy_constraint},
]

# Perform optimization
result = minimize(
    constant_potential, x_inner_initial, method="SLSQP", constraints=constraints
)

# Construct the full x vector including boundaries
x_optimized = np.concatenate(([0], result.x, [1]))
x_dot_optimized = np.gradient(x_optimized, delta_x)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot x and x_dot for initial guess and optimized results
space = np.linspace(0.0, 1.0, TOTAL_POINTS)
plt.subplot(2, 2, 1)
plt.plot(space, x_initial, label='x (Initial Guess)', color='gray', linestyle='--')
plt.plot(space, x_optimized, label='x (Optimized)', color='blue')
plt.title('x (Initial vs Optimized)')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()

# Plot L(t)
L_initial = np.gradient(x_initial, delta_x)**2
L_optimized = x_dot_optimized**2
plt.subplot(2, 2, 2)
plt.plot(space, L_initial, label='L(t) (Initial Guess)', color='gray', linestyle='--')
plt.plot(space, L_optimized, label='L(t) (Optimized)', color='orange')
plt.title('L(t) (Initial vs Optimized)')
plt.xlabel('t')
plt.ylabel('L(t)')
plt.legend()

# Plot S
plt.subplot(2, 2, 3)
plt.plot(space, np.cumsum(L_initial)/TOTAL_POINTS, label='S(t) (Initial Guess)', color='gray', linestyle='--')
plt.plot(space, np.cumsum(L_optimized)/TOTAL_POINTS, label='S(t) (Optimized)', color='orange')
plt.title('S(t) (Initial vs Optimized)')
plt.xlabel('t')
plt.ylabel('S(t)')
plt.legend()

plt.tight_layout()
plt.savefig('/app/output/output.png')
