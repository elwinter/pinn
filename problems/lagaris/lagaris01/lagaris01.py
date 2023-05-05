"""Problem definition file for simple ODE (Lagaris problem 1).

This is a first-order, nonlinear ODE, defined on the domain [0, 1], with the
initial condition Ψ(0) = 1.

This ordinary differential equation is taken from the paper:

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

    0: Ψ

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
dependent_variable_names = ["Ψ"]

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
iΨ = dependent_variable_index["Ψ"]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$\psi$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# @tf.function
def ode_Ψ(X, Y, delY):
    """Differential equation for Ψ.

    Evaluate the differential equation for Ψ.

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
    (Ψ,) = Y
    (delΨ,) = del_Y
    dΨ_dx = tf.reshape(delΨ[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dΨ_dx + (x + (1 + 3*x**2)/(1 + x + x**3))*Ψ - x**3
        - 2*x - x**2*(1 + 3*x**2)/(1 + x + x**3)
    )
    return G


# Gather the differential equations into a list.
de = [ode_Ψ]


# Parameters and functions for the analytical solution

# Original problem domain
x0 = 0.0
x1 = 1.0

# Initial condition
Ψ0 = 1.0


def Ψ_analytical(x):
    """Analytical solution to lagaris01.

    Analytical solution to lagaris01.

    Parameters
    ----------
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

    Returns
    -------
    Ψ : np.array of float, shape (n,)
        Analytical solution at each x-value.
    """
    Ψ = tf.math.exp(-x**2/2)/(1 + x + x**3) + x**2
    return Ψ


def dΨ_dx_analytical(x):
    """Analytical 1st derivative to lagaris01.

    Analytical 1st derivative of lagaris01 analytical solution.

    Parameters
    ----------
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

    Returns
    -------
    dΨ_dx : np.array of float, shape (n,)
        Value of dΨ/dx for each evaluation point.
    """
    dΨ_dx = (
        2*x - np.exp(-x**2/2)*(1 + x + 4*x**2 + x**4)/(1 + x + x**3)**2
    )
    return dΨ_dx


if __name__ == "__main__":
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")
    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    print(f"{x0} <= x <= {x1}")
    print(f"Ψ0 = {Ψ0}")

    nx = 11
    x = np.linspace(x0, x1, nx)
    Ψ = Ψ_analytical(x)
    dΨ_dx = dΨ_dx_analytical(x)
    for i in range(nx):
        print(f"{i} {x[i]} {Ψ[i]} {dΨ_dx[i]}")