"""Problem definition file for a simple 2-D MHD problem.

This problem definition file describes the 2-D current loop advection
problem, which is based on the loop2d example in the Athena MHD test suite.
Details are available at:

https://www.astro.princeton.edu/~jstone/Athena/tests/field-loop/Field-loop.html

This case deals with a line current in the +z direction (out of the
screen), with a return current toward -z at r = 0.3. +x is to the right,
+y is up.

NOTE: This version of the code solves *only* the equations for n, Bx, By, and
Bz.

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
    1: Bx (x-component of magnetic field)
    2: By (y-component of magnetic field)
    3: Bz (z-component of magnetic field)

NOTE: These equations were last verified on 2023-02-12.

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
dependent_variable_names = ["n", "Bx", "By", "Bz"]

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
i_n = dependent_variable_index["n"]
iBx = dependent_variable_index["Bx"]
iBy = dependent_variable_index["By"]
iBz = dependent_variable_index["Bz"]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$n$", "$B_x$", "$B_y$", "$B_z$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)

# Plasma parameters
A = 1e-3   # Magnitude of magnetic vector potential.
R0 = 0.3   # Radius of current cylinder.
n0 = 1.0   # Number density
P0 = 1.0   # Pressure
u0z = 0.0  # z-component of velocity
B0z = 0.0  # z-component of magnetic field

# Define the constant fluid flow field.
θ = 60.0
u0 = 1.0
u0x = u0*np.sin(np.radians(θ))
u0y = u0*np.cos(np.radians(θ))


# NOTE: In the functions defined below for the differential equations, the
# arguments can be unpacked as follows:
# def pde_XXX(X, Y, del_Y):
#     nX = X.shape[0]
#     t = tf.reshape(X[:, it], (nX, 1))
#     x = tf.reshape(X[:, ix], (nX, 1))
#     y = tf.reshape(X[:, iy], (nX, 1))
#     (n, Bx, By, Bz) = Y
#     (del_n, del_Bx, del_By, del_Bz) = del_Y
#     dn_dt = tf.reshape(del_n[:, it], (nX, 1))
#     dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
#     dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
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
    # (n, Bx, By, Bz) = Y
    (del_n, del_Bx, del_By, del_Bz) = del_Y
    dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
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
    G = dn_dt + u0x*dn_dx + u0y*dn_dy
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
    # (n, Bx, By, Bz) = Y
    (del_n, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
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
    G = dBx_dt + u0x*dBx_dx + u0y*dBx_dy
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
    # (n, Bx, By, Bz) = Y
    (del_n, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
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
    G = dBy_dt + u0x*dBy_dx + u0y*dBy_dy
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
    # (n, Bx, By, Bz) = Y
    (del_n, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dn_dy = tf.reshape(del_n[:, iy], (nX, 1))
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
    G = dBz_dt + u0x*dBz_dx + u0y*dBz_dy
    return G


# Make a list of all of the differential equations.
de = [
    pde_n,
    pde_Bx,
    pde_By,
    pde_Bz,
]


# Define analytical solutions.


def n_analytical(t, x, y):
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
    return n


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
    w = np.where(r < R0)
    Bx = np.zeros(t.shape[0])
    Bx[w] = -A*yp[w]/r[w]
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
    w = np.where(r < R0)
    By = np.zeros(t.shape[0])
    By[w] = A*xp[w]/r[w]
    return By


def Bz_analytical(t, x, y):
    """Analytical solution for the z-component of the magnetic field.

    Compute the analytical solution for the z-component of the magnetic field.

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
    n_analytical,
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
    w = np.where(r < R0)
    dBx_dx = np.zeros(t.shape[0])
    dBx_dx[w] = A*xp[w]*yp[w]/r[w]**3
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
    w = np.where(r < R0)
    dBy_dy = np.zeros(t.shape[0])
    dBy_dy[w] = -A*xp[w]*yp[w]/r[w]**3
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

    print("A = %s" % A)
    print("R0 = %s" % R0)
    print("n0 = %s" % n0)
    print("P0 = %s" % P0)
    print("u0z = %s" % u0z)
    print("B0z = %s" % B0z)

    print("θ = %s" % θ)
    print("u0 = %s" % u0)
    print("u0x = %s" % u0x)
    print("u0y = %s" % u0y)
