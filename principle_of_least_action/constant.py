import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the objective function for optimization
def objective(x_inner):
    x = np.concatenate(([0], x_inner, [1]))  # Add fixed boundary points
    x_dot = np.gradient(x)  # Compute the gradient
    return np.sum(x_dot**2)  # Minimize the sum of squares of the gradient

# Energy constraint: ensure x_dot^2 at each time step matches the initial value
def energy_constraint(x_inner):
    x = np.concatenate(([0], x_inner, [1]))
    x_dot = np.gradient(x)
    return x_dot**2 - x_dot_initial**2

# Number of interior points
n_points = 98  # Total points is n_points + 2 (boundary points)

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
    objective, x_inner_initial, method="SLSQP", constraints=constraints
)

# Construct the full x vector including boundaries
x_optimized = np.concatenate(([0], result.x, [1]))
x_dot_optimized = np.gradient(x_optimized)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot x and x_dot for initial guess and optimized results
plt.subplot(2, 2, 1)
plt.plot(x_initial, label='x (Initial Guess)', color='gray', linestyle='--')
plt.plot(x_optimized, label='x (Optimized)', color='blue')
plt.title('x (Initial vs Optimized)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# Plot L(t)
L_initial = np.gradient(x_initial)**2
L_optimized = x_dot_optimized**2
plt.subplot(2, 2, 2)
plt.plot(L_initial, label='L(t) (Initial Guess)', color='gray', linestyle='--')
plt.plot(L_optimized, label='L(t) (Optimized)', color='orange')
plt.title('L(t) (Initial vs Optimized)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# Plot S
plt.subplot(2, 2, 3)
plt.plot(np.cumsum(L_initial), label='S(t) (Initial Guess)', color='gray', linestyle='--')
plt.plot(np.cumsum(L_optimized), label='S(t) (Optimized)', color='orange')
plt.title('S(t) (Initial vs Optimized)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.savefig('/app/output/output.png')
