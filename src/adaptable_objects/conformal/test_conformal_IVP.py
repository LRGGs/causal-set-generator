import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Define the conformal factor Omega(t, x)
def Omega(t, x):
    return t ** 4 * np.exp(x**2)


def dOmega_dt(t, x):
    return 4 * t ** 3 * np.exp(x**2)


def dOmega_dx(t, x):
    return 2 * x * t ** 4 * np.exp(x**2)


# Define the geodesic equations as a system of first-order ODEs
def geodesic_equations(tau, y):
    t, x, dt_dtau, dx_dtau = y  # Correctly unpack the state vector

    omega = Omega(t, x)
    domega_dt = dOmega_dt(t, x)
    domega_dx = dOmega_dx(t, x)

    # The derivatives of our state vector components
    d2t_dtau2 = (dt_dtau ** 2 * domega_dt / 2 - dx_dtau * dt_dtau * domega_dx - domega_dx * dx_dtau * dx_dtau / 2) / omega
    d2x_dtau2 = (- dx_dtau ** 2 * domega_dx / 2 - dx_dtau * dt_dtau * domega_dt - domega_dt * dt_dtau * dt_dtau / 2) / omega

    return [dt_dtau, dx_dtau, d2t_dtau2, d2x_dtau2]


# Initial conditions: [t(0), x(0), dt/dtau(0), dx/dtau(0)]
initial_conditions = [5, 10, 1, 0.7]  # Example values

# Time span for the integration (in terms of the proper time tau)
tau_span = (0, 1)  # From tau = 0 to tau = 100

# Solve the ODE
solution = solve_ivp(geodesic_equations, tau_span, initial_conditions, method='RK45', dense_output=True)

# Use the solution
# solution.t will contain the time points
# solution.y will contain the solution [t, x, dt/dtau, dx/dtau] at each time point

# Example: Evaluate the solution at specific points
tau_eval = np.linspace(0, 1, 100)
sol_eval = solution.sol(tau_eval)

# sol_eval[0, :] will be t(tau), sol_eval[1, :] will be x(tau) at the evaluated points

# Plotting or further analysis can be done using sol_eval

plt.plot(sol_eval[1, :], sol_eval[0, :])
plt.show()
