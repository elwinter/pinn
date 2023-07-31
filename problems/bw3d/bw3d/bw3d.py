"""Problem definition file for a simple 3-D MHD problem.

This problem definition file describes a 3-D blast wave, which is based on
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
    3: z

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
independent_variable_names = ["t", "x", "y", "z"]

# Invert the independent variable list to map name to index.
independent_variable_index = {}
for (i, s) in enumerate(independent_variable_names):
    independent_variable_index[s] = i
it = independent_variable_index["t"]
ix = independent_variable_index["x"]
iy = independent_variable_index["y"]
iz = independent_variable_index["z"]

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = ["$t$", "$x$", "$y$", "$z$"]

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
    "$n$", "$P$",
    "$u_x$", "$u_y$", "$u_z$",
    "$B_x$", "$B_y$", "$B_z$"
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

# Define the constant fluid flow field.
# θ = 60.0
# u0 = 1.0
# u0x = u0*np.sin(np.radians(θ))
# u0y = u0*np.cos(np.radians(θ))


# NOTE: In the functions defined below for the differential equations, the
# arguments can be unpacked as follows:
# def pde_XXX(X, Y, del_Y):
#     nX = X.shape[0]
#     t = tf.reshape(X[:, it], (nX, 1))
#     x = tf.reshape(X[:, ix], (nX, 1))
#     y = tf.reshape(X[:, iy], (nX, 1))
#     z = tf.reshape(X[:, iz], (nX, 1))
#     (n, P, ux, uy, uz, Bx, By, Bz) = Y
#     (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
#     dn_dt = tf.reshape(del_n[:, it], (nX, 1))
#     dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
#     dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
#     dn_dz = tf.reshape(del_n[:, iz], (nX, 1))
#     dP_dt = tf.reshape(del_P[:, it], (nX, 1))
#     dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
#     dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
#     dP_dz = tf.reshape(del_P[:, iz], (nX, 1))
#     dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
#     dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
#     dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
#     dux_dz = tf.reshape(del_ux[:, iz], (nX, 1))
#     duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
#     duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
#     duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
#     duy_dz = tf.reshape(del_uy[:, iz], (nX, 1))
#     duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
#     duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
#     duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
#     duz_dz = tf.reshape(del_uz[:, iz], (nX, 1))
#     dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
#     dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
#     dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
#     dBx_dz = tf.reshape(del_Bx[:, iz], (nX, 1))
#     dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
#     dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
#     dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
#     dBy_dz = tf.reshape(del_By[:, iz], (nX, 1))
#     dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
#     dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
#     dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
#     dBz_dz = tf.reshape(del_Bz[:, iz], (nX, 1))


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
    # y = tf.reshape(X[:, iy], (nX, 1))
    # z = tf.reshape(X[:, iz], (nX, 1))
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
    dn_dz = tf.reshape(del_n[:, iz], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    # dP_dz = tf.reshape(del_P[:, iz], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    # dux_dz = tf.reshape(del_ux[:, iz], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    # duy_dz = tf.reshape(del_uy[:, iz], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    duz_dz = tf.reshape(del_uz[:, iz], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    # dBx_dz = tf.reshape(del_Bx[:, iz], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    # dBy_dz = tf.reshape(del_By[:, iz], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
    # dBz_dz = tf.reshape(del_Bz[:, iz], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dn_dt +
        n*(dux_dx + duy_dy + duz_dz) +
        dn_dx*ux + dn_dy*uy + dn_dz*uz
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
    # t = tf.reshape(X[:, it], (nX, 1))
    # x = tf.reshape(X[:, ix], (nX, 1))
    # y = tf.reshape(X[:, iy], (nX, 1))
    # z = tf.reshape(X[:, iz], (nX, 1))
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
    dn_dz = tf.reshape(del_n[:, iz], (nX, 1))
    dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    dP_dz = tf.reshape(del_P[:, iz], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    # dux_dz = tf.reshape(del_ux[:, iz], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    # duy_dz = tf.reshape(del_uy[:, iz], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    # duz_dz = tf.reshape(del_uz[:, iz], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    # dBx_dz = tf.reshape(del_Bx[:, iz], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    # dBy_dz = tf.reshape(del_By[:, iz], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
    # dBz_dz = tf.reshape(del_Bz[:, iz], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        -ɣ*P/n*(dn_dt + ux*dn_dx + uy*dn_dy +uz*dn_dz) +
        dP_dt + ux*dP_dx + uy*dP_dy + uz*dP_dz
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
    # t = tf.reshape(X[:, it], (nX, 1))
    # x = tf.reshape(X[:, ix], (nX, 1))
    # y = tf.reshape(X[:, iy], (nX, 1))
    # z = tf.reshape(X[:, iz], (nX, 1))
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
    # dn_dz = tf.reshape(del_n[:, iz], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    # dP_dz = tf.reshape(del_P[:, iz], (nX, 1))
    dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    dux_dz = tf.reshape(del_ux[:, iz], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    # duy_dz = tf.reshape(del_uy[:, iz], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    # duz_dz = tf.reshape(del_uz[:, iz], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    dBx_dz = tf.reshape(del_Bx[:, iz], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    # dBy_dz = tf.reshape(del_By[:, iz], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
    # dBz_dz = tf.reshape(del_Bz[:, iz], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        n*(dux_dt + ux*dux_dx + uy*dux_dy + uz*dux_dz) +
        (By*(dBy_dx - dBx_dy) + Bz*(dBz_dx - dBx_dz))/(m*μ0) +
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
    # t = tf.reshape(X[:, it], (nX, 1))
    # x = tf.reshape(X[:, ix], (nX, 1))
    # y = tf.reshape(X[:, iy], (nX, 1))
    # z = tf.reshape(X[:, iz], (nX, 1))
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
    # dn_dz = tf.reshape(del_n[:, iz], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    # dP_dz = tf.reshape(del_P[:, iz], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    # dux_dz = tf.reshape(del_ux[:, iz], (nX, 1))
    duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    duy_dz = tf.reshape(del_uy[:, iz], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    # duz_dz = tf.reshape(del_uz[:, iz], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    # dBx_dz = tf.reshape(del_Bx[:, iz], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    dBy_dz = tf.reshape(del_By[:, iz], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
    # dBz_dz = tf.reshape(del_Bz[:, iz], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        n*(duy_dt + ux*duy_dx + uy*duy_dy + uz*duy_dz) +
        (Bx*(dBx_dy - dBy_dx) + Bz*(dBz_dy - dBy_dz))/(m*μ0) +
         dP_dy/m
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
    # z = tf.reshape(X[:, iz], (nX, 1))
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
    # dn_dz = tf.reshape(del_n[:, iz], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    dP_dz = tf.reshape(del_P[:, iz], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    # dux_dz = tf.reshape(del_ux[:, iz], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    # duy_dz = tf.reshape(del_uy[:, iz], (nX, 1))
    duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    duz_dz = tf.reshape(del_uz[:, iz], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    dBx_dz = tf.reshape(del_Bx[:, iz], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    dBy_dz = tf.reshape(del_By[:, iz], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
    # dBz_dz = tf.reshape(del_Bz[:, iz], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        n*(duz_dt + ux*duz_dx + uy*duz_dy + uz*duz_dz) +
        (Bx*(dBx_dz - dBz_dx) + By*(dBy_dz - dBz_dy))/(m*μ0) +
        dP_dz/m
    )
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
    # z = tf.reshape(X[:, iz], (nX, 1))
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
    # dn_dz = tf.reshape(del_n[:, iz], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    # dP_dz = tf.reshape(del_P[:, iz], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    dux_dz = tf.reshape(del_ux[:, iz], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    # duy_dz = tf.reshape(del_uy[:, iz], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    duz_dz = tf.reshape(del_uz[:, iz], (nX, 1))
    dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    dBx_dz = tf.reshape(del_Bx[:, iz], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    # dBy_dz = tf.reshape(del_By[:, iz], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
    # dBz_dz = tf.reshape(del_Bz[:, iz], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dBx_dt + ux*dBx_dx + uy*dBx_dy + uz*dBx_dz +
        Bx*(duy_dy + duz_dz) - By*dux_dy - Bz*dux_dz
    )
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
    # z = tf.reshape(X[:, iz], (nX, 1))
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
    # dn_dz = tf.reshape(del_n[:, iz], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    # dP_dz = tf.reshape(del_P[:, iz], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    # dux_dz = tf.reshape(del_ux[:, iz], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    # duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    duy_dz = tf.reshape(del_uy[:, iz], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    # duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    duz_dz = tf.reshape(del_uz[:, iz], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    # dBx_dz = tf.reshape(del_Bx[:, iz], (nX, 1))
    dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    dBy_dz = tf.reshape(del_By[:, iz], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
    # dBz_dz = tf.reshape(del_Bz[:, iz], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dBy_dt + ux*dBy_dx + uy*dBy_dy + uz*dBy_dz +
        By*(dux_dx + duz_dz) - Bx*duy_dx - Bz*duy_dz
    )
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
    # z = tf.reshape(X[:, iz], (nX, 1))
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
    # dn_dz = tf.reshape(del_n[:, iz], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, iy], (nX, 1))
    # dP_dz = tf.reshape(del_P[:, iz], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, iy], (nX, 1))
    # dux_dz = tf.reshape(del_ux[:, iz], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, it], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, ix], (nX, 1))
    duy_dy = tf.reshape(del_uy[:, iy], (nX, 1))
    # duy_dz = tf.reshape(del_uy[:, iz], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, it], (nX, 1))
    duz_dx = tf.reshape(del_uz[:, ix], (nX, 1))
    duz_dy = tf.reshape(del_uz[:, iy], (nX, 1))
    # duz_dz = tf.reshape(del_uz[:, iz], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, it], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    # dBx_dz = tf.reshape(del_Bx[:, iz], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, it], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    # dBy_dz = tf.reshape(del_By[:, iz], (nX, 1))
    dBz_dt = tf.reshape(del_Bz[:, it], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
    dBz_dz = tf.reshape(del_Bz[:, iz], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dBz_dt + ux*dBz_dx + uy*dBz_dy + uz*dBz_dz +
        Bz*(dux_dx + duy_dy) - Bx*duz_dx - By*duz_dy
    )
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


if __name__ == "__main__":
    print("independent_variable_names = %s" % independent_variable_names)
    print("independent_variable_index = %s" % independent_variable_index)
    print("independent_variable_labels = %s" % independent_variable_labels)
    print("n_dim = %s" % n_dim)

    print("dependent_variable_names = %s" % dependent_variable_names)
    print("dependent_variable_index = %s" % dependent_variable_index)
    print("dependent_variable_labels = %s" % dependent_variable_labels)
    print("n_var = %s" % n_var)

    # print("A = %s" % A)
    # print("R0 = %s" % R0)
    # print("n0 = %s" % n0)
    # print("P0 = %s" % P0)
    # print("u0z = %s" % u0z)
    # print("B0z = %s" % B0z)

    # print("θ = %s" % θ)
    # print("u0 = %s" % u0)
    # print("u0x = %s" % u0x)
    # print("u0y = %s" % u0y)
