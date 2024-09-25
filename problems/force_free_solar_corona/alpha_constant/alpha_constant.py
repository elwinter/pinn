"""Problem definition for force-free solar coronal field with constant alpha.

NOTE: This is the first problem to perform parameter estimation.

This file defines the equations to solve for determining the force-free
solar coronal magnetic field given the fixed boundary conditions at x = z = 0.
The domain is 0 >= x, y, z < 50. The field is static.

The equations are:

    del dot B = 0
    del cross B = alpha*B

NOTE: The functions in this module are defined using a combination of Numpy
and TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
independent variables:

    0: x
    1: y
    2: z

NOTE: In all code, below, the following indices are assigned to physical
dependent variables describing the plasma:

    0: Bx (x-component of magnetic field)
    1: By (y-component of magnetic field)
    2: Bz (z-component of magnetic field)

NOTE: In all code, below, the following indices are assigned to equation
parameters to be determined:

    0: alpha

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.

# Import supplemental modules.
import tensorflow as tf

# Import project modules.


# Names of independent variables.
independent_variable_names = ["x", "y", "z"]

# Invert the independent variable list to map name to index.
independent_variable_index = {}
for (i, s) in enumerate(independent_variable_names):
    independent_variable_index[s] = i
ix = independent_variable_index["x"]
iy = independent_variable_index["y"]
iz = independent_variable_index["z"]

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = ["$x$", "$y$", "$z$"]

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)


# Names of dependent variables.
dependent_variable_names = ["Bx", "By", "Bz"]

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
iBx = dependent_variable_index["Bx"]
iBy = dependent_variable_index["By"]
iBz = dependent_variable_index["Bz"]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$B_x$", "$B_y$", "$B_z$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Names of equation parameters.
parameter_names = ["alpha"]

# Invert the parameter list to map name to index.
parameter_index = {}
for (i, s) in enumerate(parameter_names):
    parameter_index[s] = i
ialpha = parameter_index["alpha"]

# Labels for parameters (may use LaTex) - use for plots.
parameter_labels = [r"$\alpha$"]

# Number of parameters.
n_param = len(parameter_names)


# NOTE: In the functions defined below for the differential equations, the
# arguments can be unpacked as follows:
# def pde_XXX(X, Y, del_Y, P):
#     nX = X.shape[0]
#     x = tf.reshape(X[:, ix], (nX, 1))
#     y = tf.reshape(X[:, iy], (nX, 1))
#     z = tf.reshape(X[:, iz], (nX, 1))
#     (Bx, By, Bz) = Y
#     (del_Bx, del_By, del_Bz) = del_Y
#     (alpha,) = P
#     dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
#     dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
#     dBx_dz = tf.reshape(del_Bx[:, iz], (nX, 1))
#     dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
#     dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
#     dBy_dz = tf.reshape(del_By[:, iz], (nX, 1))
#     dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
#     dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
#     dBz_dz = tf.reshape(del_Bz[:, iz], (nX, 1))


# @tf.function
def del_cross_B_x(X, Y, del_Y, P):
    """Differential equation for the x-component of the FFMHD equation.

    Evaluate the differential equation for the x-component of the FFMHD
    equation. This equation is derived from the x-component of

        del x B = alpha*B.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_Y : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.
    P : list of n_param tf.Tensor, each shape (n, 1)
        Values of parameters at each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.

    Raises
    ------
    None
    """
    nX = X.shape[0]
    # x = tf.reshape(X[:, ix], (nX, 1))
    # y = tf.reshape(X[:, iy], (nX, 1))
    # z = tf.reshape(X[:, iz], (nX, 1))
    (Bx, By, Bz) = Y
    (del_Bx, del_By, del_Bz) = del_Y
    (alpha,) = P
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    # dBx_dz = tf.reshape(del_Bx[:, iz], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    dBy_dz = tf.reshape(del_By[:, iz], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
    # dBz_dz = tf.reshape(del_Bz[:, iz], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBz_dy - dBy_dz - alpha*Bx
    return G


# @tf.function
def del_cross_B_y(X, Y, del_Y, P):
    """Differential equation for the y-component of the FFMHD equation.

    Evaluate the differential equation for the y-component of the FFMHD
    equation. This equation is derived from the y-component of

        del x B = alpha*B.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_Y : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.
    P : list of n_param tf.Tensor, each shape (n, 1)
        Values of parameters at each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.

    Raises
    ------
    None
    """
    nX = X.shape[0]
    # x = tf.reshape(X[:, ix], (nX, 1))
    # y = tf.reshape(X[:, iy], (nX, 1))
    # z = tf.reshape(X[:, iz], (nX, 1))
    (Bx, By, Bz) = Y
    (del_Bx, del_By, del_Bz) = del_Y
    (alpha,) = P
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    dBx_dz = tf.reshape(del_Bx[:, iz], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    # dBy_dz = tf.reshape(del_By[:, iz], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
    # dBz_dz = tf.reshape(del_Bz[:, iz], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBx_dz - dBz_dx - alpha*By
    return G


# @tf.function
def del_cross_B_z(X, Y, del_Y, P):
    """Differential equation for the z-component of the FFMHD equation.

    Evaluate the differential equation for the z-component of the FFMHD
    equation. This equation is derived from the z-component of

        del x B = alpha*B.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_Y : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.
    P : list of n_param tf.Tensor, each shape (n, 1)
        Values of parameters at each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.

    Raises
    ------
    None
    """
    nX = X.shape[0]
    # x = tf.reshape(X[:, ix], (nX, 1))
    # y = tf.reshape(X[:, iy], (nX, 1))
    # z = tf.reshape(X[:, iz], (nX, 1))
    (Bx, By, Bz) = Y
    (del_Bx, del_By, del_Bz) = del_Y
    (alpha,) = P
    # dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    # dBx_dz = tf.reshape(del_Bx[:, iz], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    # dBy_dz = tf.reshape(del_By[:, iz], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
    # dBz_dz = tf.reshape(del_Bz[:, iz], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBy_dx - dBx_dy - alpha*Bz
    return G


# @tf.function
def divB(X, Y, del_Y, P):
    """Differential equation for divB for the FFMHD equation.

    Evaluate the divB of the FFMHD equation. This equation is:

        del dot B = 0.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_Y : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.
    P : list of n_param tf.Tensor, each shape (n, 1)
        Values of parameters at each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.

    Raises
    ------
    None
    """
    nX = X.shape[0]
    # x = tf.reshape(X[:, ix], (nX, 1))
    # y = tf.reshape(X[:, iy], (nX, 1))
    # z = tf.reshape(X[:, iz], (nX, 1))
    # (Bx, By, Bz) = Y
    (del_Bx, del_By, del_Bz) = del_Y
    # (alpha,) = P
    dBx_dx = tf.reshape(del_Bx[:, ix], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, iy], (nX, 1))
    # dBx_dz = tf.reshape(del_Bx[:, iz], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, ix], (nX, 1))
    dBy_dy = tf.reshape(del_By[:, iy], (nX, 1))
    # dBy_dz = tf.reshape(del_By[:, iz], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, ix], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, iy], (nX, 1))
    dBz_dz = tf.reshape(del_Bz[:, iz], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBx_dx + dBy_dy + dBz_dz
    return G


# Make a list of all of the differential equations.
de = [
    del_cross_B_x,
    del_cross_B_y,
    del_cross_B_z,
    divB,
]

# # Make a list of all of the differential equation names.
# de_names = [
#     'del_cross_B_x',
#     'del_cross_B_y',
#     'del_cross_B_z',
#     'divB',
# ]

# nde = len(de)


if __name__ == "__main__":
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_index = {independent_variable_index}")
    print(f"ix = {ix}, iy = {iy}, iz = {iz}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")

    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_index = {dependent_variable_index}")
    print(f"iBx = {iBx}, iBy = {iBy}, iBz = {iBz}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    print(f"parameter_names = {parameter_names}")
    print(f"parametere_index = {parameter_index}")
    print(f"ialpha = {ialpha}")
    print(f"parameter_labels = {parameter_labels}")
    print(f"n_param = {n_param}")
