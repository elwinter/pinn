"""Problem definition file for a simple 2-D MHD problem.

This problem definition file describes the 2-D line current advection
problem, which is based on the loop2d example in the Athena MHD test suite.
Details are available at:

https://www.astro.princeton.edu/~jstone/Athena/tests/field-loop/Field-loop.html

This case deals with a line current in the +z direction (out of the screen).
+x is to the right, +y is up.

NOTE: This version of the code solves *only* the equations for n, P, ux, uy,
uz, Bx, By, and Bz.

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

    0: P (pressure)
    1: ux (x-component of velocity)
    2: uy (y-component of velocity)
    3: uz (z-component of velocity)
    4: Bx (x-component of magnetic field)
    5: By (y-component of magnetic field)
    6: Bz (z-component of magnetic field)

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

# Invert the independent variable name list to map name to index.
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
dependent_variable_names = ["P", "ux", "uy", "uz", "Bx", "By", "Bz"]

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
iP = dependent_variable_index["P"]
iux = dependent_variable_index["ux"]
iuy = dependent_variable_index["uy"]
iuz = dependent_variable_index["uz"]
iBx = dependent_variable_index["Bx"]
iBy = dependent_variable_index["By"]
iBz = dependent_variable_index["Bz"]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$P$", "$u_x$", "$u_y$", "$u_z$", "$B_x$", "$B_y$", "$B_z$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)

# Physical constants
μ0 = 1.0  # Normalized vacuum permeability

# Plasma parameters
m = 1.0    # Plasma article mass
ɣ = 5/3    # Adiabatic index = (N + 2)/N, N = # DOF=3, not 2.
n0 = 1.0   # Number density
P0 = 1.0   # Pressure
u0z = 0.0  # z-component of velocity
B0z = 0.0  # z-component of magnetic field
I = 1e-3   # Normalized current
C1 = μ0*I/(2*np.pi)  # Leading constant in analytical solutions for Bx, By.

# Define the constant fluid flow field.
θ = 60.0  # Angle in degrees clockwise from +y axis
u0 = 1.0  # Flow speed
u0x = u0*np.sin(np.radians(θ))  # x-component of flow velocity
u0y = u0*np.cos(np.radians(θ))  # y-component of flow velocity


# Normalized physical constants.
μ0 = 1.0  # Permeability of free space
m = 1.0   # Particle mass
ɣ = 5/3   # Adiabatic index = (N + 2)/N, N = # DOF=3, not 2.


# NOTE: In the functions defined below for the differential equations, the
# arguments can be unpacked as follows:
# def pde_XXX(X, Y, del_Y):
#     nX = X.shape[0]
#     t = tf.reshape(X[:, it], (nX, 1))
#     x = tf.reshape(X[:, ix], (nX, 1))
#     y = tf.reshape(X[:, iy], (nX, 1))
#     (P, ux, uy, uz, Bx, By, Bz) = Y
#     (del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
#     dP_dt = tf.reshape(del_P[:, it], (nX, 1))
#     dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
#     dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
#     dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
#     dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
#     dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
#     duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
#     duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
#     duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
#     duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
#     duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
#     duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
#     dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
#     dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
#     dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
#     dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
#     dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
#     dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
#     dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
#     dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
#     dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))


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
    # t = tf.reshape(X[:, it], (nX, 1))
    # x = tf.reshape(X[:, ix], (nX, 1))
    # y = tf.reshape(X[:, iy], (nX, 1))
    (P, ux, uy, uz, Bx, By, Bz) = Y
    (del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dP_dt + ux*dP_dx + uy*dP_dy
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
    # t = tf.reshape(X[:, it], (nX, 1))
    # x = tf.reshape(X[:, ix], (nX, 1))
    # y = tf.reshape(X[:, iy], (nX, 1))
    (P, ux, uy, uz, Bx, By, Bz) = Y
    (del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        n0*(dux_dt + ux*dux_dx + uy*dux_dy) + dP_dx/m +
        (By*(dBy_dx - dBx_dy) + Bz*dBz_dx)/(m*μ0)
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
    # t = tf.reshape(X[:, it], (nX, 1))
    # x = tf.reshape(X[:, ix], (nX, 1))
    # y = tf.reshape(X[:, iy], (nX, 1))
    (P, ux, uy, uz, Bx, By, Bz) = Y
    (del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        n0*(duy_dt + ux*duy_dx + uy*duy_dy) + dP_dy/m +
        (Bx*(dBx_dy - dBy_dx) + Bz*dBz_dy)/(m*μ0)
    )
    return G


# @tf.function
def pde_uz(X, Y, del_Y):
    """Differential equation for z-velocity.

    Evaluate the differential equation for z-velocity. This equation is derived
    from the equation of conservation of z-momentum.

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
    # y = tf.reshape(X[:, iy], (nX, 1))
    (P, ux, uy, uz, Bx, By, Bz) = Y
    (del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = n0*(duz_dt + ux*duz_dx + uy*duz_dy) - (Bx*dBz_dx + By*dBz_dy)/(m*μ0)
    return G


# @tf.function
def pde_Bx(X, Y, del_Y):
    """Differential equation for the x-component of the magnetic field.

    Evaluate the differential equation for the x-component of the magnetic
    field. This equation is derived from the x-component of Faraday's Law.

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
    # y = tf.reshape(X[:, iy], (nX, 1))
    (P, ux, uy, uz, Bx, By, Bz) = Y
    (del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBx_dt + ux*dBx_dx + uy*dBx_dy + Bx*duy_dy - By*dux_dy
    return G


# @tf.function
def pde_By(X, Y, del_Y):
    """Differential equation for the y-component of the magnetic field.

    Evaluate the differential equation for the y-component of the magnetic
    field. This equation is derived from the y-component of Faraday's Law.

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
    # y = tf.reshape(X[:, iy], (nX, 1))
    (P, ux, uy, uz, Bx, By, Bz) = Y
    (del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBy_dt + ux*dBy_dx + uy*dBy_dy + By*dux_dx - Bx*duy_dx
    return G


# @tf.function
def pde_Bz(X, Y, del_Y):
    """Differential equation for the z-component of the magnetic field.

    Evaluate the differential equation for the z-component of the magnetic
    field. This equation is derived from the z-component of Faraday's Law.

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
    # y = tf.reshape(X[:, iy], (nX, 1))
    (P, ux, uy, uz, Bx, By, Bz) = Y
    (del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
    # G is a Tensor of shape (n, 1).
    G = (
        dBz_dt + ux*dBz_dx + uy*dBz_dy + Bz*(dux_dx + duy_dy) -
        Bx*duz_dx - By*duz_dy
    )
    return G


# Make a list of all of the differential equations.
de = [
    pde_P,
    pde_ux,
    pde_uy,
    pde_uz,
    pde_Bx,
    pde_By,
    pde_Bz,
]


# Define analytical solutions.


def P_analytical(t, x, y):
    """Analytical solution for the pressure.

    Compute the analytical solution for the pressure.

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
    P : np.array of float, shape (n,)
        Value of P for each evaluation point.
    """
    P = np.full(t.shape, P0)
    return P


def ux_analytical(t, x, y):
    """Analytical solution for the x-velocity.

    Compute the analytical solution for the x-velocity.

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
    ux : np.array of float, shape (n,)
        Value of ux for each evaluation point.
    """
    ux = np.full(t.shape, u0x)
    return ux


def uy_analytical(t, x, y):
    """Analytical solution for the y-velocity.

    Compute the analytical solution for the y-velocity.

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
    uy : np.array of float, shape (n,)
        Value of uy for each evaluation point.
    """
    uy = np.full(t.shape, u0y)
    return uy


def uz_analytical(t, x, y):
    """Analytical solution for the z-velocity.

    Compute the analytical solution for the z-velocity.

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
    uz : np.array of float, shape (n,)
        Value of uz for each evaluation point.
    """
    uz = np.full(t.shape, u0z)
    return uz


def Bx_analytical(t, x, y):
    """Analytical solution for the x-component of the magnetic field.

    Compute the analytical solution for the x-component of the magnetic field.
    (xp, yp) are the coordinates (x, y) translated back to the initial frame
    for field computation, since the analytical solution is a simple linear
    translation of the initial conditions.

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
    Bx : np.array of float, shape (n,)
        Value of Bx for each evaluation point.
    """
    xp = x - u0x*t
    yp = y - u0y*t
    r = np.sqrt(xp**2 + yp**2)
    Bx = -C1*yp/r**2
    return Bx


def By_analytical(t, x, y):
    """Analytical solution for the y-component of the magnetic field.

    Compute the analytical solution for the y-component of the magnetic field.
    (xp, yp) are the coordinates (x, y) translated back to the initial frame
    for field computation, since the analytical solution is a simple linear
    translation of the initial conditions.

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
    By : np.array of float, shape (n,)
        Value of By for each evaluation point.
    """
    xp = x - u0x*t
    yp = y - u0y*t
    r = np.sqrt(xp**2 + yp**2)
    By = C1*xp/r**2
    return By


def Bz_analytical(t, x, y):
    """Analytical solution for the z-component of the magnetic field.

    Compute the analytical solution for the z-component of the magnetic field.
    (xp, yp) are the coordinates (x, y) translated back to the initial frame
    for field computation, since the analytical solution is a simple linear
    translation of the initial conditions.

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
    Bz : np.array of float, shape (n,)
        Value of Bz for each evaluation point.
    """
    Bz = np.full(t.shape, B0z)
    return Bz


# Gather the analytical solutions in a list.
# Use same order as dependent_variable_names.
analytical_solutions = [
    P_analytical,
    ux_analytical,
    uy_analytical,
    uz_analytical,
    Bx_analytical,
    By_analytical,
    Bz_analytical,
]


# Other useful analytical functions.


def dBx_dx_analytical(t, x, y):
    """Analytical solution for dBx/dx of the magnetic field.

    Compute the analytical solution for dBx/dx of the magnetic field.
    (xp, yp) are the coordinates (x, y) translated back to the initial frame
    for field computation, since the analytical solution is a simple linear
    translation of the initial conditions.

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
    dBx_dx : np.array of float, shape (n,)
        Value of dBx/dx for each evaluation point.
    """
    xp = x - u0x*t
    yp = y - u0y*t
    r = np.sqrt(xp**2 + yp**2)
    dBx_dx = 2*C1*xp*yp/r**4
    return dBx_dx


def dBy_dy_analytical(t, x, y):
    """Analytical solution for dBy/dy of the magnetic field.

    Compute the analytical solution for dBy/dy of the magnetic field.
    (xp, yp) are the coordinates (x, y) translated back to the initial frame
    for field computation, since the analytical solution is a simple linear
    translation of the initial conditions.

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
    dBy_dy : np.array of float, shape (n,)
        Value of dBy/dy for each evaluation point.
    """
    xp = x - u0x*t
    yp = y - u0y*t
    r = np.sqrt(xp**2 + yp**2)
    dBy_dy = -2*C1*xp*yp/r**4
    return dBy_dy


if __name__ == "__main__":
    print("independent_variable_names = %s" % independent_variable_names)
    print("independent_variable_index = %s" % independent_variable_index)
    print("independent_variable_labels = %s" % independent_variable_labels)
    print("n_dim = %s" % n_dim)

    print("dependent_variable_names = %s" % dependent_variable_names)
    print("dependent_variable_index = %s" % dependent_variable_index)
    print("dependent_variable_labels = %s" % dependent_variable_labels)
    print("n_var = %s" % n_var)

    print("μ0 = %s" % μ0)
    print("m = %s" % m)
    print("ɣ = %s" % ɣ)
    print("n0 = %s" % n0)
    print("P0 = %s" % P0)
    print("u0z = %s" % u0z)
    print("B0z = %s" % B0z)
    print("I = %s" % I)
    print("C1 = %s" % C1)

    print("θ = %s" % θ)
    print("u0 = %s" % u0)
    print("u0x = %s" % u0x)
    print("u0y = %s" % u0y)
