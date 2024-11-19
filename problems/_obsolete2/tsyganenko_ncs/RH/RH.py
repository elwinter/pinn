"""Problem definition file for RH.

This file describes Equation (4) for RH from:

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
independent_variable_labels = ["$f_P$", "$f_{Bz}$", r"$\phi$"]

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ['RH']

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
iRH = dependent_variable_index['RH']

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$R_H$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Make the empirical function from the top-level module available in
# this namespace.
RH_empirical = tncs.RH_empirical


if __name__ == '__main__':
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")
    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    # Test the analytical solution.
    n = 11
    fPmin, fPmax, nfP = -2.0, 2.0, n
    fBzmin, fBzmax, nfBz = -2.0, 2.0, n
    phimin, phimax, nphi = 0.0, 0.0, n
    fP = np.linspace(fPmin, fPmax, nfP)
    fBz = np.linspace(fBzmin, fBzmax, nfBz)
    phi = np.linspace(phimin, phimax, nphi)
    RH = RH_empirical(fP, fBz, phi)
    for i in range(n):
        print(f"{i} {fP[i]} {fBz[i]} {phi[i]} {RH[i]}")
