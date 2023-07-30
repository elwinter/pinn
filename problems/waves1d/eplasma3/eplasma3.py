"""Problem definition file for a simple 1-D MHD problem.

This problem definition file describes an electron plasma wave: unit pressure
and density. The wave has 3 frequency components.

This problem specifies the initial condition, and the boundary condition at
x = 0, so the result is a wave train propagating in t and x.

This problem uses the linearized MHD equations.

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
independent variables:

    0: t
    1: x

NOTE: In all code, below, the following indices are assigned to physical
dependent variables (these are 1st-order perturbation values):

    0:  n1 (number density)
    1:  u1x (x-component of velocity)
    2:  E1x (x-component of electric field)

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.

# Import supplemental modules.
import numpy as np
import tensorflow as tf

# Import project modules.
from pinn import plasma


# Names of independent variables.
independent_variable_names = ["t", "x"]

# Invert the independent variable name list to map name to index.
independent_variable_index = {}
for (i, s) in enumerate(independent_variable_names):
    independent_variable_index[s] = i
it = independent_variable_index["t"]
ix = independent_variable_index["x"]

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = ["$t$", "$x$"]

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = [
    "n1", "u1x", "E1x"
]

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
i_n1 = dependent_variable_index["n1"]
iu1x = dependent_variable_index["u1x"]
iE1x = dependent_variable_index["E1x"]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = [
    "$n_1$", "$u_{1x}$", "$E_{1x}$"
]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Normalized physical constants.
e = 1.0     # Unit charge
me = 1.0    # Electron mass
kb = 1.0    # Boltzmann constant
eps0 = 1.0  # Permittivity of free space


# Plasma parameters
n0 = 1.0  # Ambient equilibrium number density
P0 = 1.0  # Ambient equilibrium pressure
ɣ = 5/3   # Adiabatic index = (N + 2)/N, N = # DOF
T = 1.0   # Ambient temperature

# Wavelength and wavenumber of initial n/vx/Ex perturbations.
wavelengths = np.array([0.5, 1.0, 2.0])
kx = 2*np.pi/wavelengths
nc = len(kx)  # Number of wave components.

# Steady-state value and perturbation amplitudes for number density.
n0 = 1.0
n1_amp = np.array([0.1, 0.1, 0.1])

# Compute the electron plasma wave angular frequency for each component.                           
w = plasma.electron_plasma_wave_angular_frequency(n0, T, kx, normalize=True)

# Steady-state value and perturbation amplitudes for x-velocity.
u1x0 = 0.0
u1x_amp = w/kx*n1_amp/n0

# Steady-state value and perturbation amplitudes for x-electric field.
E1x0 = 0.0
E1x_amp = e*n1_amp/(kx*eps0)


# @tf.function
def pde_n1(X, Y, del_Y):
    """Differential equation for number density perturbation.

    Evaluate the differential equation for number density perturbation. This
    equation is derived from the linearized equation of mass continuity.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_Y : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables
        at each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    nX = X.shape[0]
    # t = tf.reshape(X[:, it], (nX, 1))
    # x = tf.reshape(X[:, ix], (nX, 1))
    # (n1, u1x, E1x) = Y
    (del_n1, del_u1x, del_E1x) = del_Y
    dn1_dt = tf.reshape(del_n1[:, it], (nX, 1))
    # dn1_dx = tf.reshape(del_n1[:, ix], (nX, 1))
    # du1x_dt = tf.reshape(del_u1x[:, it], (nX, 1))
    du1x_dx = tf.reshape(del_u1x[:, ix], (nX, 1))
    # dE1x_dt = tf.reshape(del_E1x[:, it], (nX, 1))
    # dE1x_dx = tf.reshape(del_E1x[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dn1_dt + n0*du1x_dx
    return G


# @tf.function
def pde_u1x(X, Y, del_Y):
    """Differential equation for x-velocity perturbation.

    Evaluate the differential equation for x-velocity perturbation. This
    equation is derived from the linearized equation of conservation of
    x-momentum.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_Y : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    n = X.shape[0]
    # t = tf.reshape(X[:, 0], (n, 1))
    # x = tf.reshape(X[:, 1], (n, 1))
    (n1, u1x, E1x) = Y
    (del_n1, del_u1x, del_E1x) = del_Y
    # dn1_dt = tf.reshape(del_n1[:, 0], (n, 1))
    dn1_dx = tf.reshape(del_n1[:, 1], (n, 1))
    du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
    # du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
    # dE1x_dt = tf.reshape(del_E1x[:, 0], (n, 1))
    # dE1x_dx = tf.reshape(del_E1x[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = du1x_dt + e/me*E1x + ɣ*kb*T/(me*n0)*dn1_dx
    return G


# @tf.function
def pde_E1x(X, Y, del_Y):
    """Differential equation for x-electric field perturbation.

    Evaluate the differential equation for x-electric field perturbation.
    This equation is derived from the linearized x-component of ???.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_Y : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    n = X.shape[0]
    # t = tf.reshape(X[:, 0], (n, 1))
    # x = tf.reshape(X[:, 1], (n, 1))
    (n1, u1x, E1x) = Y
    (del_n1, del_u1x, del_E1x) = del_Y
    # dn1_dt = tf.reshape(del_n1[:, 0], (n, 1))
    # dn1_dx = tf.reshape(del_n1[:, 1], (n, 1))
    # du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
    # du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
    # dE1x_dt = tf.reshape(del_E1x[:, 0], (n, 1))
    dE1x_dx = tf.reshape(del_E1x[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dE1x_dx + e/eps0*n1
    return G


# Make a list of all of the differential equations.
de = [
    pde_n1,
    pde_u1x,
    pde_E1x
]


# Define analytical solutions.


def n1_analytical(X: np.ndarray):
    """Compute analytical solution for number density perturbation.

    Compute anaytical solution for number density perturbation.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    n1 : np.ndarray of float, shape (n,)
        Analytical values for number density perturbation.
    """
    t = X[:, it]
    x = X[:, ix]
    n1 = np.zeros_like(x)
    for (n1i, ki, wi) in zip(n1_amp, kx, w):
        n1 += n1i*np.sin(ki*x - wi*t)
    return n1


def u1x_analytical(X: np.ndarray):
    """Compute analytical solution for x-velocity perturbation.

    Compute anaytical solution for x-velocity perturbation.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    u1x : np.ndarray of float, shape (n,)
        Analytical values for x-velocity perturbation.
    """
    t = X[:, it]
    x = X[:, ix]
    u1x = np.zeros_like(x)
    for (u1xi, ki, wi) in zip(u1x_amp, kx, w):
        u1x += u1xi*np.sin(ki*x - wi*t)
    return u1x


def E1x_analytical(X: np.ndarray):
    """Compute analytical solution for x-electric field perturbation.

    Compute anaytical solution for x-electric field perturbation.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    E1x : np.ndarray of float, shape (n,)
        Analytical values for x-electric field perturbation.
    """
    t = X[:, it]
    x = X[:, ix]
    E1x = np.zeros_like(x)
    for (E1xi, ki, wi) in zip(E1x_amp, kx, w):
        E1x += E1xi*np.sin(ki*x - wi*t + np.pi/2)
    return E1x


# Gather all of the analytical solutions into a list.
analytical_solutions = [
    n1_analytical,
    u1x_analytical,
    E1x_analytical,
]


if __name__ == "__main__":
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_index = {independent_variable_index}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")

    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_index = {dependent_variable_index}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    print(f"e = {e}")
    print(f"me = {me}")
    print(f"kb = {kb}")
    print(f"eps0 = {eps0}")

    print(f"n0 = {n0}")
    print(f"P0 = {P0}")
    print(f"ɣ = {ɣ}")
    print(f"T = {T}")

    print(f"wavelengths = {wavelengths}")
    print(f"kx = {kx}")
    print(f"w = {w}")

    print(f"n0 = {n0}")
    print(f"n1_amp = {n1_amp}")

    print(f"w = {w}")

    print(f"u1x0 = {u1x0}")
    print(f"u1x_amp = {u1x_amp}")

    print(f"E1x0 = {E1x0}")
    print(f"E1x_amp = {E1x_amp}")
