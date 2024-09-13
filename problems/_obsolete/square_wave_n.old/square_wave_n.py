"""Problem definition file for a simple 1-D square wave advection problem.

This problem definition file describes the 1-D square wave advection
problem, which was suggested in:

J. P. Boris and D. L. Book, J. Comput. Phys. 11, 38 (1973).

NOTE: This version of the code solves *only* the equation for n.

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
dependent_variable_names = ["n"]

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
i_n = dependent_variable_index["n"]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$n$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)

# Flow parameters
u0x = 1.0  # x-component of flow velocity
n0 = 0.5   # Ambient number density
n_wave = 2.0   # Square wave number density
x0, x1 = 0.01, 0.21  # Boundaries of initial square wave


# NOTE: In the functions defined below for the differential equations, the
# arguments can be unpacked as follows:
# def pde_XXX(X, Y, del_Y):
#     nX = X.shape[0]
#     t = tf.reshape(X[:, it], (nX, 1))
#     x = tf.reshape(X[:, ix], (nX, 1))
#     (n,) = Y
#     (del_n,) = del_Y
#     dn_dt = tf.reshape(del_n[:, it], (nX, 1))
#     dn_dx = tf.reshape(del_n[:, ix], (nX, 1))


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
    # t = tf.reshape(X[:, it], (nX, 1))
    # x = tf.reshape(X[:, ix], (nX, 1))
    # (n,) = Y
    (del_n,) = del_Y
    dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    dn_dx = tf.reshape(del_n[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dn_dt + u0x*dn_dx
    return G


# Make a list of all of the differential equations.
de = [
    pde_n,
]


# Define analytical solutions.


def n_analytical(t, x):
    """Analytical solution for the number density.

    Compute the analytical solution for the number density.

    Parameters
    ----------
    t : np.array of float, shape (n,)
        Value of t for each evaluation point.
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.
    y : np.array of float, shape (n,)
        Value of y for each evaluation point.

    Returns
    -------
    n : np.array of float, shape (n,)
        Value of n for each evaluation point.
    """
    n = np.full(t.shape, n0)
    ge = x >= x0 + u0x*t
    le = x <= x1 + u0x*t
    mask = np.logical_and(ge, le)
    n[mask] = n_wave
    return n


# Gather the analytical solutions in a list.
# Use same order as dependent_variable_names.
analytical_solutions = [
    n_analytical,
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

    print("n0 = %s" % n0)
    print("u0x = %s" % u0x)
