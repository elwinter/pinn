"""Problem definition file for simple ODE parameter solve (Lagaris problem 1).

This is a first-order, nonlinear ODE, with the initial condition Ψ(0) = 1.

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

NOTE: In all code, below, the following indices are assigned to equation
parameters to be determined:

    0: c1

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
dependent_variable_labels = [r"$\psi$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Names of equation parameters.
parameter_names = ["c1"]

# Invert the parameter list to map name to index.
parameter_index = {}
for (i, s) in enumerate(parameter_names):
    parameter_index[s] = i
ic1 = parameter_index["c1"]

# Labels for parameters (may use LaTex) - use for plots.
parameter_labels = [r"$c_1"]

# Number of parameters.
n_param = len(parameter_names)


# @tf.function
def ode_Ψ(X, Y, delY, P):
    """Differential equation for Ψ.

    Evaluate the ordinary differential equation for Ψ(x).

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    delY : list of n_var tf.Tensor, each shape (n, n_dim)
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
    # x is a Tensor of shape (nX, 1).
    x = tf.reshape(X[:, ix], (nX, 1))
    # Ψ is a Tensor of shape (nX, 1).
    (Ψ,) = Y
    # delΨ is a Tensor of shape (nX, 1).
    (delΨ,) = delY
    (c1,) = P
    # dΨ_dx is a Tensor of shape (nX, 1).
    dΨ_dx = tf.reshape(delΨ[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dΨ_dx + (x + (1 + 3*x**2)/(1 + x + x**3))*Ψ - x**3
        - c1*x - x**2*(1 + 3*x**2)/(1 + x + x**3)
    )
    return G


# Gather the differential equations into a list.
de = [ode_Ψ]


# Initial condition for the analytical solution
Ψ0 = 1.0


def Ψ_analytical(x):
    """Analytical solution to  lagaris01p.

    Analytical solution to lagaris01.

    Parameters
    ----------
    x : np.array of float, shape (n,)
        Value of x for each evaluation point.

    Returns
    -------
    Ψ : np.array of float, shape (n,)
        Analytical solution at each x-value.

    Raises
    ------
    None
    """
    Ψ = np.exp(-x**2/2)/(1 + x + x**3) + x**2
    return Ψ


# Gather analytical solutions into a list.
Y_analytical = [
    Ψ_analytical
]

# def dΨ_dx_analytical(x):
#     """Analytical 1st derivative to lagaris01.

#     Analytical 1st derivative of lagaris01 analytical solution.

#     Parameters
#     ----------
#     x : np.array of float, shape (n,)
#         Value of x for each evaluation point.

#     Returns
#     -------
#     dΨ_dx : np.array of float, shape (n,)
#         Value of dΨ/dx for each evaluation point.

#     Raises
#     ------
#     None
#     """
#     dΨ_dx = (
#         2*x - np.exp(-x**2/2)*(1 + x + 4*x**2 + x**4)/(1 + x + x**3)**2
#     )
#     return dΨ_dx


if __name__ == "__main__":
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_index = {independent_variable_index}")
    print(f"ix = {ix}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")

    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_index = {dependent_variable_index}")
    print(f"iΨ = {iΨ}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    print(f"parameter_names = {parameter_names}")
    print(f"parametere_index = {parameter_index}")
    print(f"ic1 = {ic1}")
    print(f"parameter_labels = {parameter_labels}")
    print(f"n_param = {n_param}")
