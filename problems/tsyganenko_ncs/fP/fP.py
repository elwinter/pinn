"""Problem definition file for fP.

This file describes Equation (10) for fP from:

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
independent_variable_names = ["P"]

# Invert the independent variable list to map name to index.
independent_variable_index = {}
for (i, s) in enumerate(independent_variable_names):
    independent_variable_index[s] = i
iP = independent_variable_index["P"]

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = ["$P$"]

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ["fP"]

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
ifP = dependent_variable_index["fP"]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$f_P$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Scale and exponent for P.
Pmean = 2.0  # nPa
chi = -0.29  # Table 1


def fP_analytical(P):
    """Analytical form for fP.

    Analytical form for fP.

    Parameters
    ----------
    P : np.array of float, shape (n,)
        Value of P for each evaluation point.

    Returns
    -------
    fP : np.array of float, shape (n,)
        Analytical value of fP at each evaluation point.

    Raises
    ------
    None
    """
    fP = (P/Pmean)**chi - 1
    return fP


if __name__ == "__main__":
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")
    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    # Test the analytical solution and derivative.
    Pmin, Pmax, nP = 0.0, 10.0, 11
    P = np.linspace(Pmin, Pmax, nP)
    fP = fP_analytical(P)
    for i in range(nP):
        print(f"{i} {P[i]} {fP[i]}")
