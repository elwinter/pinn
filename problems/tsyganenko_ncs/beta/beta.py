"""Problem definition file for beta.

This file describes Equation (9) for beta from:

N. A. Tsyganenko, V. A. Andreeva, and E. I. Gordeev, "Internally and
externally induced deformations of the magnetospheric equatorial current as
inferred from spacecraft data", Ann. Geophys., 33, 1–11, 2015

www.ann-geophys.net/33/1/2015/
doi:10.5194/angeo-33-1-2015

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.

# Import supplemental modules.
import numpy as np

# Import project modules.
import problems.tsyganenko_ncs.tsyganenko_ncs as tncs


# Names of independent variables.
independent_variable_names = ['fBz']

# Invert the independent variable list to map name to index.
independent_variable_index = {}
for (i, s) in enumerate(independent_variable_names):
    independent_variable_index[s] = i
ifBz = independent_variable_index['fBz']

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = ["$f_{Bz}$"]

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ['beta']

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
ibeta = dependent_variable_index['beta']

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$\beta$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Make the empirical function from the top-level module available in
# this namespace.
beta_empirical = tncs.beta_empirical


if __name__ == '__main__':
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")
    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    # Test the empirical equation.
    fBzmin, fBzmax, nfBz = -2.0, 2.0, 21
    fBz = np.linspace(fBzmin, fBzmax, nfBz)
    beta = beta_empirical(fBz)
    for i in range(nfBz):
        print(f"{i} {fBz[i]} {beta[i]}")
