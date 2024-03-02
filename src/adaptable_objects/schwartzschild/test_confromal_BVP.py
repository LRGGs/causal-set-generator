import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Boundary condition variables
t_initial, x_initial = 5, 10  # Initial coordinates
t_final, x_final = 10, 10  # Final coordinates


# Define the conformal factor Omega(t, x) and its derivatives
def Omega(t, x):
    return x * t


def dOmega_dt(t, x):
    return x


def dOmega_dx(t, x):
    return t


# Geodesic equations for BVP: corrected to properly handle the state vector y
def geodesic_equations_bvp(tau, y):
    t, x, dt_dtau, dx_dtau = y  # Correctly unpack the state vector

    omega = Omega(t, x)
    domega_dt = dOmega_dt(t, x)
    domega_dx = dOmega_dx(t, x)

    # The derivatives of our state vector components
    d2t_dtau2 = (dt_dtau ** 2 * domega_dt / 2 - dx_dtau * dt_dtau * domega_dx - domega_dx * dx_dtau * dx_dtau / 2) / omega
    d2x_dtau2 = (- dx_dtau ** 2 * domega_dx / 2 - dx_dtau * dt_dtau * domega_dt - domega_dt * dt_dtau * dt_dtau / 2) / omega
    return np.vstack((dt_dtau, dx_dtau, d2t_dtau2, d2x_dtau2))


# Boundary conditions function using the variables
def boundary_conditions(ya, yb):
    # Correct to only compare positions, not velocities
    return np.array([ya[0] - t_initial, ya[1] - x_initial, yb[0] - t_final, yb[1] - x_final])


# Initial mesh of points where the solution is approximated
tau = np.linspace(0, 10, 1000)

# Initial guess for the solution at these points
# Adjusted to include guesses for both positions and their derivatives
y_guess = np.zeros((4, tau.size))  # 4 for t, x, dt/dtau, dx/dtau
y_guess[0] = np.linspace(t_initial, t_final, tau.size)  # Guess for t
y_guess[1] = np.linspace(x_initial, x_final, tau.size)  # Guess for x
#y_guess[2] = np.array([1 for i in tau])
#y_guess[3] = np.array([1e-2 if i < 5 else -1e-2 for i in tau])
#print(y_guess)
# Assuming initial guesses for derivatives (dt/dtau, dx/dtau) as zeros
# Modify these guesses if you have better estimates

# Solve the BVP
solution = solve_bvp(geodesic_equations_bvp, boundary_conditions, tau, y_guess)

# Check if the solution was successful and evaluate it
if solution.success:
    tau_eval = np.linspace(0, 10, 1000)
    t_eval, x_eval = solution.sol(tau_eval)[:2]  # Extract t and x solutions
    plt.plot(x_eval, t_eval)
    plt.show()
    print("Solution found!")
else:
    print("Solution not found. Consider adjusting the initial guess or the mesh.")
