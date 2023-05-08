"""Problem definition file for pair of coupled ODE (Lagaris problem 4).

This is a first-order, nonlinear ODE, defined on the domain [0, 3], with the
initial conditions Ψ1(0) = 0,  Ψ2(0) = 1.

This problem is taken from the paper:

"Artificial Neural Networks for Solving Ordinary and Partial Differential
Equations", by Isaac Elias Lagaris, Aristidis Likas, and Dimitrios I. Fotiadis
IEEE Transactions on Neural Networks, Vol. 9, No. 5, September 1998 987

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
independent variables:

    0: x

NOTE: In all code, below, the following indices are assigned to physical
dependent variables:

    0: Ψ1
    1: Ψ2

NOTE: For all methods:

X represents an a set of arbitrary evaluation points. It is a tf.Tensor with
shape (n, n_dim), where n is the number of evaluation points, and n_dim is
the number of dimensions (independent variables) in the problem. For an ODE,
n_dim is 1, giving a shape of (n, 1).

Y represents a set of dependent variables at each point in X. This variable is
a list of n_var tf.Tensor, each shape (n, 1), where n_var is the number of
dependent variables. For an ODE, n_var is 1, giving a list of 1 Tensor of
shape of (n, 1).

delY contains the first derivatives of each dependent variable with respect
to each independent variable, at each point in X. It is a list of n_var
tf.Tensor, each shape (n, n_dim). For an ODE, n_var and n_dim are 1, for a
list of 1 Tensor of shape (n, 1).

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
independent_variable_names = ["x"]

# Invert the independent variable list to map name to index.
independent_variable_index = {}
for (i, s) in enumerate(independent_variable_names):
    independent_variable_index[s] = i
ix = independent_variable_index["x"]

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = ["$x$"]

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ["Ψ1", "Ψ2"]

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
iΨ1 = dependent_variable_index["Ψ1"]
iΨ2 = dependent_variable_index["Ψ2"]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$\psi_1$", "$\psi_2$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# @tf.function
def ode_Ψ1(X, Y, delY):
    """Differential equation for Ψ1.

    Evaluate the differential equation for Ψ1.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    delY : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    nX = X.shape[0]
    x = tf.reshape(X[:, ix], (nX, 1))
    (Ψ1, Ψ2) = Y
    (delΨ1, delΨ2) = delY
    dΨ1_dx = tf.reshape(delΨ1[:, ix], (nX, 1))
    # dΨ2_dx = tf.reshape(delΨ2[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = G = dΨ1_dx - tf.math.cos(x) - Ψ1**2 - Ψ2 + 1 + x**2 + tf.math.sin(x)**2
    return G


# @tf.function
def ode_Ψ2(X, Y, delY):
    """Differential equation for Ψ2.

    Evaluate the differential equation for Ψ2.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    delY : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    nX = X.shape[0]
    x = tf.reshape(X[:, ix], (nX, 1))
    (Ψ1, Ψ2) = Y
    (delΨ1, delΨ2) = delY
    # dΨ1_dx = tf.reshape(delΨ1[:, ix], (nX, 1))
    dΨ2_dx = tf.reshape(delΨ2[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = G = dΨ2_dx - 2*x + (1 + x**2)*tf.math.sin(x) - Ψ1*Ψ2
    return G


# Gather the differential equations into a list.
de = [ode_Ψ1, ode_Ψ2]


# Parameters and functions for the analytical solution

# Original problem domain
x0 = 0.0
x1 = 3.0

# Initial conditions
Ψ10 = 0.0
Ψ20 = 1.0


def Ψ1_analytical(x):
    """Analytical solution to lagaris04 equation 1.

    Analytical solution to  lagaris04 equation 1.

    Parameters
    ----------
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

    Returns
    -------
    Ψ1 : np.array of float, shape (n,)
        Analytical solution at each x-value.
    """
    Ψ1 = np.sin(x)
    return Ψ1


def Ψ2_analytical(x):
    """Analytical solution to lagaris04 equation 2.

    Analytical solution to  lagaris04 equation 2.

    Parameters
    ----------
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

    Returns
    -------
    Ψ2 : np.array of float, shape (n,)
        Analytical solution at each x-value.
    """
    Ψ2 = 1 + x**2
    return Ψ2


def dΨ1_dx_analytical(x):
    """Analytical 1st derivative to lagaris04 equation 1.

    Analytical 1st derivative of lagaris04 equation 1.

    Parameters
    ----------
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

    Returns
    -------
    dΨ1_dx : np.array of float, shape (n,)
        Value of dΨ1/dx for each evaluation point.
    """
    dΨ1_dx = np.cos(x)
    return dΨ1_dx


def dΨ2_dx_analytical(x):
    """Analytical 1st derivative to lagaris04 equation 2.

    Analytical 1st derivative of lagaris04 equation 2.

    Parameters
    ----------
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

    Returns
    -------
    dΨ2_dx : np.array of float, shape (n,)
        Value of dΨ2/dx for each evaluation point.
    """
    dΨ2_dx = 2*x
    return dΨ2_dx


if __name__ == "__main__":
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")
    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    print(f"{x0} <= x <= {x1}")
    print(f"Ψ10 = {Ψ10}")
    print(f"Ψ20 = {Ψ20}")

    nx = 11
    x = np.linspace(x0, x1, nx)
    Ψ1 = Ψ1_analytical(x)
    Ψ2 = Ψ2_analytical(x)
    dΨ1_dx = dΨ1_dx_analytical(x)
    dΨ2_dx = dΨ2_dx_analytical(x)
    for i in range(nx):
        print(f"{i} {x[i]} {Ψ1[i]} {Ψ2[i]} {dΨ1_dx[i]} {dΨ2_dx[i]}")