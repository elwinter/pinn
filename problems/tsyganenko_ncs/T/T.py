"""Problem definition file for T.

This file describes Equation (5) for T from:

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
independent_variable_names = ['fP', 'fBz', 'phi']

# Invert the independent variable list to map name to index.
independent_variable_index = {}
for (i, s) in enumerate(independent_variable_names):
    independent_variable_index[s] = i
ifP = independent_variable_index['fP']
ifBz = independent_variable_index['fBz']
iphi = independent_variable_index['phi']

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = [r"$f_P$", r"$f_{Bz}", r"$\phi$"]

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ['T']

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
iT = dependent_variable_index['T']

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = [r"$T$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Make the empirical function from the top-level module available in
# this namespace.
T_empirical = tncs.T_empirical


if __name__ == '__main__':
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")
    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    # Test the empirical equation.
    fPmin, fPmax, nfP = -1.0, 2.0, 21
    fBz_ = 0.0
    phi_ = 0.0
    fP = np.linspace(fPmin, fPmax, nfP)
    fBz = np.full(fP.shape, fBz_)
    phi = np.full(fP.shape, phi_)
    T = T_empirical(fP, fBz, phi)
    for i in range(nfP):
        print(f"{i} {fP[i]} {T[i]}")
