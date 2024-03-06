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
dependent_variable_names = ['T']

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
iT = dependent_variable_index['T']

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$T$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Empirical constants for T equation, and RMS mean absolute deviation
T0, T0_rms = 0.29, 0.02
T1, T1_rms = 0.18, 0.08


def T_analytical(fP, fBz, phi):
    """Analytical form for T.

    Analytical form for T.

    Parameters
    ----------
    fp : np.array of float, shape (n,)
        Values of fp for each evaluation point.
    fBz : np.array of float, shape (n,)
        Values of fBz for each evaluation point.
    phi : np.array of float, shape (n,)
        Values of phi for each evaluation point.

    Returns
    -------
    T : np.array of float, shape (n,)
        Analytical value of T at each evaluation point.

    Raises
    ------
    None
    """
    T = T0 + T1*fP
    return T


if __name__ == "__main__":
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")
    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    # Test the analytical solution.
    n = 11
    fPmin, fPmax, nfP = -2.0, 2.0, n
    fP = np.linspace(fPmin, fPmax, nfP)
    fBzmin, fBzmax, nfBz = -2.0, 2.0, n
    fBz = np.linspace(fBzmin, fBzmax, nfBz)
    phimin, phimax, nphi = 0.0, 2*np.pi, n
    phi = np.linspace(phimin, phimax, nphi)
    T = T_analytical(fP, fBz, phi)
    for i in range(n):
        print(f"{i} {fP[i]} {fBz[i]} {phi[i]} {T[i]}")
