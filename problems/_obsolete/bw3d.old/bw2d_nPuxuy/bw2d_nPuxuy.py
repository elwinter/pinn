"""Problem definition file for a simple 2-D MHD problem.

This problem definition file describes a 2-D blast wave, which is based on
the 2-D blast wave example in the Athena MHD test suite. Details are
available at:

https://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
independent variables:

    0: t
    1: x
    2: y

NOTE: In all code, below, the following indices are assigned to physical
dependent variables:

    0: n (number density)
    1: P (pressure)
    2: ux (x-component of velocity)
    3: uy (y-component of velocity)

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.

# Import supplemental modules.
import numpy as np
import tensorflow as tf

# Import project modules.


# Names of independent variables.
independent_variable_names = ["t", "x", "y"]

# Invert the independent variable list to map name to index.
independent_variable_index = {}
for (i, s) in enumerate(independent_variable_names):
    independent_variable_index[s] = i
it = independent_variable_index["t"]
ix = independent_variable_index["x"]
iy = independent_variable_index["y"]

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = ["$t$", "$x$", "$y$"]

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ["n", "P", "ux", "uy"]

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
i_n = dependent_variable_index["n"]
iP = dependent_variable_index["P"]
iux = dependent_variable_index["ux"]
iuy = dependent_variable_index["uy"]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = [
    "$n$", "$P$",
    "$u_x$", "$u_y$"
]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Plasma parameters
# R0 = 0.1   # Radius of blast sphere
# n0 = 1.0   # Number density
# P0 = 0.1   # Ambient pressure at t = 0 outside R0
# u0x = 0.0  # Initial x-component of velocity
# u0r = 0.0  # Initial y-component of velocity
# u0z = 0.0  # Initial z-component of velocity
# B0x = 1/np.sqrt(2)  # Initial Bx
# B0y = 1/np.sqrt(2)  # Initial Bz
# B0z = 0.0  # Initial Bz
ɣ = 5/3    # Adiabatic index = (N + 2)/N, N = # DOF=3, not 2.
μ0 = 1.0  # Normalized vacuum permeability
m = 1.0    # Plasma article mass


# NOTE: In the functions defined below for the differential equations, the
# arguments can be unpacked as follows:
# def pde_XXX(X, Y, del_Y):
#     nX = X.shape[0]
#     t = tf.reshape(X[:, it], (nX, 1))
#     x = tf.reshape(X[:, ix], (nX, 1))
#     y = tf.reshape(X[:, iy], (nX, 1))
#     (n, P, ux, uy) = Y
#     (del_n, del_P, del_ux, del_uy) = del_Y
#     dn_dt = tf.reshape(del_n[:, it], (nX, 1))
#     dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
#     dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
#     dP_dt = tf.reshape(del_P[:, it], (nX, 1))
#     dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
#     dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
#     dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
#     dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
#     dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
#     duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
#     duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
#     duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))


# @tf.function
def pde_n(X, Y, del_Y):
    """Differential equation for number density.

    Evaluate the differential equation for number density. This equation is
    derived from the equation of mass continuity.

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
#     t = tf.reshape(X[:, it], (nX, 1))
#     x = tf.reshape(X[:, ix], (nX, 1))
#     y = tf.reshape(X[:, iy], (nX, 1))
    (n, P, ux, uy) = Y
    (del_n, del_P, del_ux, del_uy) = del_Y
    dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
#     dP_dt = tf.reshape(del_P[:, it], (nX, 1))
#     dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
#     dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
#     dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
#     dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
#     duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
#     duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dn_dt +
        n*(dux_dx + duy_dy) +
        dn_dx*ux + dn_dy*uy
    )
    return G


# @tf.function
def pde_P(X, Y, del_Y):
    """Differential equation for P.

    Evaluate the differential equation for pressure (or energy density). This
    equation is derived from the equation of conservation of energy.

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
#     t = tf.reshape(X[:, it], (nX, 1))
#     x = tf.reshape(X[:, ix], (nX, 1))
#     y = tf.reshape(X[:, iy], (nX, 1))
    (n, P, ux, uy) = Y
    (del_n, del_P, del_ux, del_uy) = del_Y
    dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
    dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
#     dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
#     dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
#     dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
#     duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
#     duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
#     duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        -ɣ*P/n*(dn_dt + ux*dn_dx + uy*dn_dy) +
        dP_dt + ux*dP_dx + uy*dP_dy
    )
    return G


# @tf.function
def pde_ux(X, Y, del_Y):
    """Differential equation for x-velocity.

    Evaluate the differential equation for x-velocity. This equation is derived
    from the equation of conservation of x-momentum.

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
    nX = X.shape[0]
#     t = tf.reshape(X[:, it], (nX, 1))
#     x = tf.reshape(X[:, ix], (nX, 1))
#     y = tf.reshape(X[:, iy], (nX, 1))
    (n, P, ux, uy) = Y
    (del_n, del_P, del_ux, del_uy) = del_Y
#     dn_dt = tf.reshape(del_n[:, it], (nX, 1))
#     dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
#     dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
#     dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
#     dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
#     duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
#     duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
#     duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        n*(dux_dt + ux*dux_dx + uy*dux_dy) +
        dP_dx/m
    )
    return G


# @tf.function
def pde_uy(X, Y, del_Y):
    """Differential equation for y-velocity.

    Evaluate the differential equation for y-velocity. This equation is derived
    from the equation of conservation of y-momentum.

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
    nX = X.shape[0]
#     t = tf.reshape(X[:, it], (nX, 1))
#     x = tf.reshape(X[:, ix], (nX, 1))
#     y = tf.reshape(X[:, iy], (nX, 1))
    (n, P, ux, uy) = Y
    (del_n, del_P, del_ux, del_uy) = del_Y
#     dn_dt = tf.reshape(del_n[:, it], (nX, 1))
#     dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
#     dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
#     dP_dt = tf.reshape(del_P[:, it], (nX, 1))
#     dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
#     dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
#     dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
#     dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        n*(duy_dt + ux*duy_dx + uy*duy_dy) +
         dP_dy/m
    )
    return G


# Make a list of all of the differential equations.
de = [
    pde_n,
    pde_P,
    pde_ux,
    pde_uy,
]


if __name__ == "__main__":
    print("independent_variable_names = %s" % independent_variable_names)
    print("independent_variable_index = %s" % independent_variable_index)
    print("independent_variable_labels = %s" % independent_variable_labels)
    print("n_dim = %s" % n_dim)

    print("dependent_variable_names = %s" % dependent_variable_names)
    print("dependent_variable_index = %s" % dependent_variable_index)
    print("dependent_variable_labels = %s" % dependent_variable_labels)
    print("n_var = %s" % n_var)
