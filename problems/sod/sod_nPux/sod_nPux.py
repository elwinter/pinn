"""Problem definition file for a 1-D Sod shock tube.

This problem is described at:

http://wonka.physics.ncsu.edu/pub/VH-1/bproblems.php

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
independent variables:

    0: t
    1: x

NOTE: In all code, below, the following indices are assigned to physical
dependent variables:

    0: n (number density)
    1: P (pressure)
    2: ux (x-component of velocity)

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.

# Import supplemental modules.
import tensorflow as tf

# Import project modules.


# Names of independent variables.
independent_variable_names = ["t", "x"]

# Invert the independent variable list to map name to index.
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
dependent_variable_names = ["n", "P", "ux"]

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
i_n = dependent_variable_index["n"]
iP = dependent_variable_index["P"]
iux = dependent_variable_index["ux"]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$n$", "$P$", "$u_x$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Plasma parameters
ɣ = 1.4    # Adiabatic index = Cp/Cv
μ0 = 1.0   # Normalized vacuum permeability
m = 1.0    # Plasma article mass


# NOTE: In the functions defined below for the differential equations, the
# arguments can be unpacked as follows:
# def pde_XXX(X, Y, del_Y):
#     nX = X.shape[0]
#     t = tf.reshape(X[:, it], (nX, 1))
#     x = tf.reshape(X[:, ix], (nX, 1))
#     (n, P, ux) = Y
#     (del_n, del_P, del_ux) = del_Y
#     dn_dt = tf.reshape(del_n[:, it], (nX, 1))
#     dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
#     dP_dt = tf.reshape(del_P[:, it], (nX, 1))
#     dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
#     dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
#     dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))


# @tf.function
def pde_n(X, Y, del_Y):
    """Differential equation for number density (n).

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
    # t = tf.reshape(X[:, it], (nX, 1))
    # x = tf.reshape(X[:, ix], (nX, 1))
    (n, P, ux) = Y
    (del_n, del_P, del_ux) = del_Y
    dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dn_dt + n*dux_dx + dn_dx*ux
    return G


# @tf.function
def pde_P(X, Y, del_Y):
    """Differential equation for thermal pressure (P).

    Evaluate the differential equation for pressure. This equation is derived
    from the equation of conservation of energy.

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
    (n, P, ux) = Y
    (del_n, del_P, del_ux) = del_Y
    dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = -ɣ*P/n*(dn_dt + ux*dn_dx) + dP_dt + ux*dP_dx
    return G


# @tf.function
def pde_ux(X, Y, del_Y):
    """Differential equation for x-velocity (ux).

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
    # t = tf.reshape(X[:, it], (nX, 1))
    # x = tf.reshape(X[:, ix], (nX, 1))
    (n, P, ux) = Y
    (del_n, del_P, del_ux) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = n*(dux_dt + ux*dux_dx) + dP_dx/m
    return G


# Make a list of all of the differential equations.
de = [
    pde_n,
    pde_P,
    pde_ux,
]


if __name__ == "__main__":
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_index = {independent_variable_index}")
    print(f"it = {it}")
    print(f"ix = {ix}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")

    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_index = {dependent_variable_index}")
    print(f"i_n = {i_n}")
    print(f"iP = {iP}")
    print(f"iux = {iux}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    print(f"ɣ = {ɣ}")
    print(f"μ0 = {μ0}")
    print(f"m = {m}")
