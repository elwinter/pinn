"""Problem definition file for Zs.

This file describes Equation (3) for Zs from:

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
independent_variable_names = ['rho', 'phi', 'P', 'By', 'Bz', 'psi']

# Invert the independent variable list to map name to index.
independent_variable_index = {}
for (i, s) in enumerate(independent_variable_names):
    independent_variable_index[s] = i
irho = independent_variable_index['rho']
iphi = independent_variable_index['phi']
iP = independent_variable_index['P']
iBy = independent_variable_index['By']
iBz = independent_variable_index['Bz']
ipsi = independent_variable_index['psi']

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = [
    r"$\rho$", r"$\phi$", "$P$", "$B_y$", "$B_z", r"$\psi$"
]

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ['Zs']

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
iZs = dependent_variable_index['Zs']

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$Z_s$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Make the empirical function from the top-level module available in
# this namespace.
Zs_empirical = tncs.Zs_empirical


if __name__ == '__main__':
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")
    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    # Test the analytical solution.
    # Values are for Figure 5.
    n = 11
    # rhomin, rhomax, nrho = 1.0, 11.0, n
    # phimin, phimax, nphi = 0.0, 2*np.pi, n
    # Pmin, Pmax, nP = 2.0, 2.0, n
    # Bymin, Bymax, nBy = 0.0, 0.0, n
    # Bzmin, Bzmax, nBz = 0.0, 0.0, n
    # psimin, psimax, npsi = np.radians(30.0), np.radians(30.0), n
    rhomin, rhomax, nrho = 1.0, 11.0, n
    phimin, phimax, nphi = np.pi, np.pi, n
    Pmin, Pmax, nP = 2.0, 2.0, n
    Bymin, Bymax, nBy = 0.0, 0.0, n
    Bzmin, Bzmax, nBz = 0.0, 0.0, n
    psimin, psimax, npsi = np.radians(30.0), np.radians(30.0), n

    rho = np.linspace(rhomin, rhomax, nrho)
    phi = np.linspace(phimin, phimax, nphi)
    P = np.linspace(Pmin, Pmax, nP)
    By = np.linspace(Bymin, Bymax, nBy)
    Bz = np.linspace(Bzmin, Bzmax, nBz)
    psi = np.linspace(psimin, psimax, npsi)

    Zs = Zs_empirical(rho, phi, P, By, Bz, psi)

    for i in range(n):
        print(f"{i} {rho[i]} {phi[i]} {P[i]} {By[i]} {Bz[i]} {psi[i]} {Zs[i]}")
