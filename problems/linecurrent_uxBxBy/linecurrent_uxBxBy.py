"""Problem definition file for a simple 2-D MHD problem.

This problem definition file describes the 2-D line current convection
problem, which is based on the loop2d example in the Athena MHD test suite.
Details are available at:

https://www.astro.princeton.edu/~jstone/Athena/tests/field-loop/Field-loop.html

NOTE: This case deals only with a line current in the +z direction (out of
the screen). +x is to the right, +y is up.

NOTE: This version of the code solves *only* the equations for ux, Bx, and By.

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

    0: ux (x-component of velocity)
    1: Bx (x-component of magnetic field)
    2: By (y-component of magnetic field)

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
independent_variable_names = ['t', 'x', 'y']

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = ["$t$", "$x$", "$y$"]

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ['ux', 'Bx', 'By']

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$u_x$", "$B_x$", "$B_y$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Normalized physical constants.
μ0 = 1.0  # Permeability of free space
m = 1.0   # Particle mass
ɣ = 5/3   # Adiabatic index = (N + 2)/N, N = # DOF=3, not 2.

# Define the constant fluid flow field.
Q = 60.0
u0 = 1.0
u0x = u0*np.sin(np.radians(Q))
u0y = u0*np.cos(np.radians(Q))

# Constant values
n = 1.0
P = 1.0
Bz = 0.0


# NOTE: In the functions defined below for the differential equations, the
# arguments can be unpacked as follows:
# def pde_XXX(X, Y, del_Y):
#     nX = X.shape[0]
#     t = tf.reshape(X[:, 0], (nX, 1))
#     x = tf.reshape(X[:, 1], (nX, 1))
#     y = tf.reshape(X[:, 2], (nX, 1))
#     (ux, Bx, By) = Y
#     (del_ux, del_Bx, del_By) = del_Y
#     dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
#     dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
#     dux_dy = tf.reshape(del_ux[:, 2], (nX, 1))
#     dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
#     dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
#     dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
#     dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
#     dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
#     dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))


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
    # t = tf.reshape(X[:, 0], (nX, 1))
    # x = tf.reshape(X[:, 1], (nX, 1))
    # y = tf.reshape(X[:, 2], (nX, 1))
    (ux, Bx, By) = Y
    (del_ux, del_Bx, del_By) = del_Y
    dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    dux_dy = tf.reshape(del_ux[:, 2], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        n*(dux_dt + ux*dux_dx + u0y*dux_dy) + By*(dBy_dx - dBx_dy)/(m*μ0)
    )
    return G


# @tf.function
def pde_Bx(X, Y, del_Y):
    """Differential equation for x-magnetic field.

    Evaluate the differential equation for x-magnetic field. This equation is
    derived from the x-component of Faraday's Law.

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
    # t = tf.reshape(X[:, 0], (nX, 1))
    # x = tf.reshape(X[:, 1], (nX, 1))
    # y = tf.reshape(X[:, 2], (nX, 1))
    (ux, Bx, By) = Y
    (del_ux, del_Bx, del_By) = del_Y
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    dux_dy = tf.reshape(del_ux[:, 2], (nX, 1))
    dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBx_dt + ux*dBx_dx + u0y*dBx_dy - By*dux_dy
    return G


# @tf.function
def pde_By(X, Y, del_Y):
    """Differential equation for y-magnetic field.

    Evaluate the differential equation for y-magnetic field. This equation is
    derived from the y-component of Faraday's Law.

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
    # t = tf.reshape(X[:, 0], (nX, 1))
    # x = tf.reshape(X[:, 1], (nX, 1))
    # y = tf.reshape(X[:, 2], (nX, 1))
    (ux, Bx, By) = Y
    (del_ux, del_Bx, del_By) = del_Y
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, 2], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
    dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBy_dt + ux*dBy_dx + u0y*dBy_dy + By*dux_dx
    return G


# Make a list of all of the differential equations.
de = [
    pde_ux,
    pde_Bx,
    pde_By,
]


if __name__ == "__main__":
    print("independent_variable_names = %s" % independent_variable_names)
    print("independent_variable_labels = %s" % independent_variable_labels)
    print("n_dim = %s" % n_dim)
    print("dependent_variable_names = %s" % dependent_variable_names)
    print("dependent_variable_labels = %s" % dependent_variable_labels)
    print("n_var = %s" % n_var)

    print("μ0 = %s" % μ0)
    print("m = %s" % m)
    print("ɣ = %s" % ɣ)

    print("Q = %s" % Q)
    print("u0 = %s" % u0)
    print("ux = %s" % u0x)
    print("uy = %s" % u0y)
