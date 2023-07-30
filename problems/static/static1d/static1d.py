"""Problem definition file for a simple 1-D MHD problem.

This problem definition file describes a static situation: unit pressure and
density, all else is 0. The run should just converge to the initial values for
each quantity.

NOTE: This version of the code solves the complete set of equations for
n, P, ux, uy, uz, Bx, By, and Bz.

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
    3: uy (y-component of velocity)
    4: uz (z-component of velocity)
    5: Bx (x-component of magnetic field)
    6: By (y-component of magnetic field)
    7: Bz (z-component of magnetic field)

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
dependent_variable_names = ["n", "P", "ux", "uy", "uz", "Bx", "By", "Bz"]

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
i_n = dependent_variable_index["n"]
iP = dependent_variable_index["P"]
iux = dependent_variable_index["ux"]
iuy = dependent_variable_index["uy"]
iuz = dependent_variable_index["uz"]
iBx = dependent_variable_index["Bx"]
iBy = dependent_variable_index["By"]
iBz = dependent_variable_index["Bz"]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = [
    "$n$", "$P$", "$u_x$", "$u_y$", "$u_z$", "$B_x$", "$B_y$", "$B_z$"
]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Plasma parameters
m = 1.0    # Plasma article mass
ɣ = 5/3    # Adiabatic index = (N + 2)/N, N = # DOF=3, not 2.


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
#     (n, P, ux, uy, uz, Bx, By, Bz) = Y
#     (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
#     dn_dt = tf.reshape(del_n[:, it], (nX, 1))
#     dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
#     dP_dt = tf.reshape(del_P[:, it], (nX, 1))
#     dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
#     dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
#     dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
#     duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
#     duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
#     duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
#     duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
#     dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
#     dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
#     dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
#     dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
#     dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
#     dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))


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
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dn_dt + n*dux_dx + dn_dx*ux
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
    # t = tf.reshape(X[:, it], (nX, 1))
    # x = tf.reshape(X[:, ix], (nX, 1))
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = -ɣ*P/n*(dn_dt + ux*dn_dx) + dP_dt + ux*dP_dx
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
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = n*(dux_dt + ux*dux_dx) + dP_dx/m + (By*dBy_dx + Bz*dBz_dx)/(m*μ0)
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
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = n*(duy_dt + ux*duy_dx) - Bx*dBy_dx/(m*μ0)
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
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = n*(duz_dt + ux*duz_dx) - Bx*dBz_dx/(m*μ0)
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
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBx_dt + ux*dBx_dx
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
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBy_dt + ux*dBy_dx + By*dux_dx - Bx*duy_dx
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
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBz_dt + ux*dBz_dx + Bz*dux_dx - Bx*duz_dx
    return G


# Make a list of all of the differential equations.
de = [
    pde_n,
    pde_P,
    pde_ux,
    pde_uy,
    pde_uz,
    pde_Bx,
    pde_By,
    pde_Bz,
]


# Define analytical solutions.

# Unit density and pressure, no velocity or magnetic field.
n0 = 1.0
P0 = 1.0
u0x = 0.0
u0y = 0.0
u0z = 0.0
B0x = 0.0
B0y = 0.0
B0z = 0.0

def n_analytical(t, x):
    """Analytical solution for the number density.

    Compute the analytical solution for the number density.

    Parameters
    ----------
    t : np.array of float, shape (n,)
        Value of t for each evaluation point.
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

    Returns
    -------
    n : np.array of float, shape (n,)
        Value of n for each evaluation point.
    """
    n = np.full(t.shape, n0)
    return n


def P_analytical(t, x):
    """Analytical solution for the pressure.

    Compute the analytical solution for the pressure.

    Parameters
    ----------
    t : np.array of float, shape (n,)
        Value of t for each evaluation point.
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

    Returns
    -------
    P : np.array of float, shape (n,)
        Value of P for each evaluation point.
    """
    P = np.full(t.shape, P0)
    return P


def ux_analytical(t, x):
    """Analytical solution for the x-velocity.

    Compute the analytical solution for the x-velocity.

    Parameters
    ----------
    t : np.array of float, shape (n,)
        Value of t for each evaluation point.
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

    Returns
    -------
    ux : np.array of float, shape (n,)
        Value of ux for each evaluation point.
    """
    ux = np.full(t.shape, u0x)
    return ux


def uy_analytical(t, x):
    """Analytical solution for the y-velocity.

    Compute the analytical solution for the y-velocity.

    Parameters
    ----------
    t : np.array of float, shape (n,)
        Value of t for each evaluation point.
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

    Returns
    -------
    uy : np.array of float, shape (n,)
        Value of uy for each evaluation point.
    """
    uy = np.full(t.shape, u0y)
    return uy


def uz_analytical(t, x):
    """Analytical solution for the z-velocity.

    Compute the analytical solution for the z-velocity.

    Parameters
    ----------
    t : np.array of float, shape (n,)
        Value of t for each evaluation point.
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

    Returns
    -------
    uz : np.array of float, shape (n,)
        Value of uz for each evaluation point.
    """
    uz = np.full(t.shape, u0z)
    return uz


def Bx_analytical(t, x):
    """Analytical solution for the x-component of the magnetic field.

    Compute the analytical solution for the x-component of the magnetic field.

    Parameters
    ----------
    t : np.array of float, shape (n,)
        Value of t for each evaluation point.
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

    Returns
    -------
    Bx : np.array of float, shape (n,)
        Value of Bx for each evaluation point.
    """
    Bx = np.full(t.shape, B0x)
    return Bx


def By_analytical(t, x):
    """Analytical solution for the y-component of the magnetic field.

    Compute the analytical solution for the y-component of the magnetic field.

    Parameters
    ----------
    t : np.array of float, shape (n,)
        Value of t for each evaluation point.
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

    Returns
    -------
    By : np.array of float, shape (n,)
        Value of By for each evaluation point.
    """
    By = np.full(t.shape, B0y)
    return By


def Bz_analytical(t, x):
    """Analytical solution for the z-component of the magnetic field.

    Compute the analytical solution for the z-component of the magnetic field.

    Parameters
    ----------
    t : np.array of float, shape (n,)
        Value of t for each evaluation point.
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

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
    n_analytical,
    P_analytical,
    ux_analytical,
    uy_analytical,
    uz_analytical,
    Bx_analytical,
    By_analytical,
    Bz_analytical,
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

    print(f"μ0 = {μ0}")
    print(f"m = {m}")
    print(f"ɣ = {ɣ}")

    print(f"n0 = {n0}")
    print(f"P0 = {P0}")
    print(f"u0x = {u0x}")
    print(f"u0y = {u0y}")
    print(f"u0z = {u0z}")
    print(f"B0x = {B0x}")
    print(f"B0y = {B0y}")
    print(f"B0z = {B0z}")
