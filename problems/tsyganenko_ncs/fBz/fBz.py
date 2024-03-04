"""Problem definition file for fBz.

This file describes Equation (10) for fBz from:

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
# import tensorflow as tf

# Import project modules.


# Names of independent variables.
independent_variable_names = ["Bz"]

# Invert the independent variable list to map name to index.
independent_variable_index = {}
for (i, s) in enumerate(independent_variable_names):
    independent_variable_index[s] = i
iBz = independent_variable_index["Bz"]

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = [r"$B_z$ (nT)"]

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ["fBz"]

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
ifBz = dependent_variable_index["fBz"]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = [r"$f_{Bz}$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Scale for Bz.
Bz0 = 5.0  # nT

def fBz_analytical(Bz):
    """Analytical form for fBz.

    Analytical form for fBz.

    Parameters
    ----------
    Bz : np.array of float, shape (n,)
        Value of Bz for each evaluation point.

    Returns
    -------
    fBz : np.array of float, shape (n,)
        Analytical value of fBz at each evaluation point.

    Raises
    ------
    None
    """
    fBz = Bz/Bz0
    return fBz


if __name__ == "__main__":
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")
    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    # Test the analytical solution and derivative.
    Bzmin, Bzmax, nBz = -2.0, 2.0, 11
    Bz = np.linspace(Bzmin, Bzmax, nBz)
    fBz = fBz_analytical(Bz)
    for i in range(nBz):
        print(f"{i} {Bz[i]} {fBz[i]}")
