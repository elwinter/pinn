"""Problem definition file for wave_equation_1d.

This file describes a simple wave equation of the form:

y(x, t) = A*sin(k*x - w*t + phi)

where:

A = 1
k = 1
w = 1
phi = pi/2

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.

# Import supplemental modules.
import numpy as np

# Import project modules.


# Names of independent variables.
independent_variable_names = ['t', 'x']

# Invert the independent variable list to map name to index.
independent_variable_index = {}
for (i, s) in enumerate(independent_variable_names):
    independent_variable_index[s] = i
it = independent_variable_index['t']
ix = independent_variable_index['x']

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = ["$t$", "$x$"]

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ['y']

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
iy = dependent_variable_index['y']

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$y$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Constants
A = 1.0
k = 1.0
w = 1.0
phi = np.pi/2

def y_analytical(t, x):
    y = A*np.sin(k*x - w*t + phi)
    return y


if __name__ == '__main__':
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")
    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    # Test the empirical equation.
    n = 11
    tmin, tmax, nt = 0.0, 1.0, n
    xmin, xmax, nx = -2.0, 2.0, n
    t = np.linspace(tmin, tmax, nt)
    x = np.linspace(xmin, xmax, nx)
    y = y_analytical(t, x)
    for i in range(n):
        print(f"{i} {t[i]} {x[i]} {y[i]}")
