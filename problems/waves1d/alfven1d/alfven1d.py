"""Problem definition file for a simple 1-D MHD problem.

This problem definition file describes an Alfven wave: unit pressure and
density, with a constant axial magnetic field (B0x = constant).

This problem specifies the initial condition, and the boundary condition at
x = 0, so the result is a wave train propagating in t and x.

This problem uses the linearized MHD equations.

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
independent variables:

    0: t
    1: x

NOTE: In all code, below, the following indices are assigned to physical
dependent variables (these are 1st-order perturbation values):

    0: n1 (number density)
    1: P1 (pressure)
    2: u1x (x-component of velocity)
    3: u1y (y-component of velocity)
    4: u1z (z-component of velocity)
    5: B1x (x-component of magnetic field)
    6: B1y (y-component of magnetic field)
    7: B1z (z-component of magnetic field)

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.

# Import supplemental modules.
import numpy as np
import tensorflow as tf

# Import project modules.
# from nn1dmhd import plasma
# from nn1dmhd.training_data import create_training_points_gridded


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
dependent_variable_names = [
    "n1", "P1",
    "u1x", "u1y", "u1z",
    "B1x", "B1y", "B1z"
]

# Invert the dependent variable list to map name to index.
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
i_n1 = dependent_variable_index["n1"]
iP1 = dependent_variable_index["P1"]
iu1x = dependent_variable_index["u1x"]
iu1y = dependent_variable_index["u1y"]
iu1z = dependent_variable_index["u1z"]
iB1x = dependent_variable_index["B1x"]
iB1y = dependent_variable_index["B1y"]
iB1z = dependent_variable_index["B1z"]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = [
    "$n_1$", "$P_1$",
    "$u_{1x}$", "$u_{1y}$", "$u_{1z}$",
    "$B_{1x}$", "$B_{1y}$", "$B_{1z}$"
]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Normalized physical constants.
# 𝑒 = 1.0     # Unit charge
# 𝑘b = 1.0    # Boltzmann constant
# ε0 = 1.0  # Permeability of free space
# 𝑚e = 1.0    # Electron mass
# μ0 = 1.0  # Permeability of free space

# Define the problem domain.
# t0 = 0.0
# t1 = 1.0
# x0 = 0.0
# x1 = 1.0
# domain = np.array(
#     [[t0, t1],
#      [x0, x1]]
# )


# Pasma parameters
n0 = 1.0  # Ambient equilibrium number density
P0 = 1.0  # Ambient equilibrium pressure
ɣ = 5/3   # Adiabatic index = (N + 2)/N, N = # DOF

# Ambient temperature (normalized to unit physical constants).
# T = 1.0

# Adiabatic index = (N + 2)/N, N = # DOF.

# Initial values for each dependent variable (dimensionless).
# ρ0 = 1.0
# P0 = 1.0
# u0x = 0.0
# u0y = 0.0
# u0z = 0.0
# B0x = 1.0
# B0y = 0.0
# B0z = 0.0
# initial_values = [ρ0, P0, u0x, u0y, u0z, B0x, B0y, B0z]


# Plasma computed values

# Compute the electron thermal speed (independent of components).
# vth = plasma.electron_thermal_speed(T, normalize=True)

# Compute the electron plasma angular frequency (independent of components).
# ωp = plasma.electron_plasma_angular_frequency(ρ0, normalize=True)

# Alfven speed.
# C_Alfven = B0x/np.sqrt(ρ0)


# Perturbations

# Perturbation amplitudes for each dependent variable (dimensionless).
# ρ1_amp = 0.0
# P1_amp = 0.0
# u1x_amp = 0.0
# u1y_amp = 0.1
# u1z_amp = 0.0
# B1x_amp = 0.0
# B1y_amp = 0.1
# B1z_amp = 0.0

# Wavelength and wavenumber of initial perturbations.
# λ = 1.0
# kx = 2*np.pi/λ

# Compute the electron plasma wave angular frequency for each component.
# ω = plasma.electron_plasma_wave_angular_frequency(ρ0, T, kx, normalize=True)

# Compute the wave phase speed for each component.
# vphase = plasma.electron_plasma_wave_phase_speed(ρ0, T, kx, normalize=True)

# Frequency and angular frequency of initial perturbation.
# f = C_Alfven/λ
# ω = 2*np.pi*f


# def create_training_data_gridded(nt: int, nx: int):
#     """Create the training data on an evenly-spaced grid.

#     Create and return a set of training points evenly spaced in x, and
#     t. Flatten the data to a list of points. Also return copies of the data
#     containing only internal points, and only boundary points.

#     Boundary points occur where:

#     t = t0

#     Parameters
#     ----------
#     nt, nx : int
#         Number of points in t-, and x-dimensions.

#     Returns
#     -------
#     Xg : np.ndarray, shape (nt*nx, n_dim)
#         Array of all training points.
#     Xg_in : np.ndarray, shape (n_in, n_dim)
#         Array of all training points within the boundary.
#     Xg_bc : np.ndarray, shape (n_bc, n_dim)
#         Array of all training points on the boundary.
#     """
#     # Create the training grid and mask.
#     ng = [nt, nx]
#     ng_total = np.prod(ng)
#     Xg = create_training_points_gridded(ng, domain)
#     mask = np.ones(ng_total, dtype=bool)

#     # Compute the coordinates of each training point, and in-domain mask value.
#     for (i, (t, x)) in enumerate(Xg):
#         if np.isclose(t, t0) or np.isclose(x, x0):
#             # This is a boundary point - mask it out.
#             mask[i] = False

#     # Flatten the mask.
#     mask.shape = (ng_total,)

#     # Extract the internal points.
#     Xg_in = Xg[mask]

#     # Invert the mask and extract the boundary points.
#     mask = np.logical_not(mask)
#     Xg_bc = Xg[mask]

#     # Return the complete, inner, and boundary training points.
#     return Xg, Xg_in, Xg_bc


# def compute_boundary_conditions(X: np.ndarray):
#     """Compute the boundary conditions.

#     The boundary conditions are computed using the analytical solutions at the
#     specified points. Note that this can be used to compute synthetic data
#     values within the domain.

#     Parameters
#     ----------
#     X : np.ndarray of float, shape (n_bc, n_dim)
#         Independent variable values for computation.

#     Returns
#     -------
#     bc : np.ndarray of float, shape (n_bc, n_var)
#         Values of each dependent variable at X.
#     """
#     n = len(X)
#     bc = np.empty((n, n_var))
#     for (i, ψa) in enumerate(ψ_analytical):
#         bc[:, i] = ψa(X)
#     return bc


# @tf.function
def pde_n1(X, Y, del_Y):
    """Differential equation for number density perturbation.

    Evaluate the differential equation for number density perturbation. This
    equation is derived from the linearized equation of mass continuity.

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
    (n1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = Y
    (del_n1, del_P1,
     del_u1x, del_u1y, del_u1z,
     del_B1x, del_B1y, del_B1z) = del_Y
    dn1_dt = tf.reshape(del_n1[:, it], (nX, 1))
    # dn1_dx = tf.reshape(del_n1[:, ix], (nX, 1))
    # dP1_dt = tf.reshape(del_P1[:, it], (nX, 1))
    # dP1_dx = tf.reshape(del_P1[:, ix], (nX, 1))
    # du1x_dt = tf.reshape(del_u1x[:, it], (nX, 1))
    du1x_dx = tf.reshape(del_u1x[:, ix], (nX, 1))
    # du1y_dt = tf.reshape(del_u1y[:, it], (nX, 1))
    # du1y_dx = tf.reshape(del_u1y[:, ix], (nX, 1))
    # du1z_dt = tf.reshape(del_u1z[:, it], (nX, 1))
    # du1z_dx = tf.reshape(del_u1z[:, ix], (nX, 1))
    # dB1x_dt = tf.reshape(del_B1x[:, it], (nX, 1))
    # dB1x_dx = tf.reshape(del_B1x[:, ix], (nX, 1))
    # dB1y_dt = tf.reshape(del_B1y[:, it], (nX, 1))
    # dB1y_dx = tf.reshape(del_B1y[:, ix], (nX, 1))
    # dB1z_dt = tf.reshape(del_B1z[:, it], (nX, 1))
    # dB1z_dx = tf.reshape(del_B1z[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dn1_dt + n0*du1x_dx
    return G


# @tf.function
def pde_P1(X, Y, del_Y):
    """Differential equation for pressure perturbation.

    Evaluate the differential equation for pressure perturbation. This
    equation is derived from the linearized equation of conservation of
    energy.

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
    (n1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = Y
    (del_n1, del_P1,
     del_u1x, del_u1y, del_u1z,
     del_B1x, del_B1y, del_B1z) = del_Y
    dn1_dt = tf.reshape(del_n1[:, it], (nX, 1))
    # dn1_dx = tf.reshape(del_n1[:, ix], (nX, 1))
    dP1_dt = tf.reshape(del_P1[:, it], (nX, 1))
    # dP1_dx = tf.reshape(del_P1[:, ix], (nX, 1))
    # du1x_dt = tf.reshape(del_u1x[:, it], (nX, 1))
    # du1x_dx = tf.reshape(del_u1x[:, ix], (nX, 1))
    # du1y_dt = tf.reshape(del_u1y[:, it], (nX, 1))
    # du1y_dx = tf.reshape(del_u1y[:, ix], (nX, 1))
    # du1z_dt = tf.reshape(del_u1z[:, it], (nX, 1))
    # du1z_dx = tf.reshape(del_u1z[:, ix], (nX, 1))
    # dB1x_dt = tf.reshape(del_B1x[:, it], (nX, 1))
    # dB1x_dx = tf.reshape(del_B1x[:, ix], (nX, 1))
    # dB1y_dt = tf.reshape(del_B1y[:, it], (nX, 1))
    # dB1y_dx = tf.reshape(del_B1y[:, ix], (nX, 1))
    # dB1z_dt = tf.reshape(del_B1z[:, it], (nX, 1))
    # dB1z_dx = tf.reshape(del_B1z[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dP1_dt - ɣ*P0/n0*dn1_dt
    return G


# # @tf.function
# def pde_u1x(X, ψ, del_ψ):
#     """Differential equation for x-velocity perturbation.

#     Evaluate the differential equation for x-velocity perturbation. This
#     equation is derived from the equation of conservation of x-momentum.

#     Parameters
#     ----------
#     X : tf.Variable, shape (n, n_dim)
#         Values of independent variables at each evaluation point.
#     ψ : list of n_var tf.Tensor, each shape (n, 1)
#         Values of dependent variables at each evaluation point.
#     del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
#         Values of gradients of dependent variables wrt independent variables at
#         each evaluation point.

#     Returns
#     -------
#     G : tf.Tensor, shape (n, 1)
#         Value of differential equation at each evaluation point.
#     """
#     n = X.shape[0]
#     t = tf.reshape(X[:, 0], (n, 1))
#     x = tf.reshape(X[:, 1], (n, 1))
#     (ρ1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = ψ
#     (del_ρ1, del_P1, del_u1x, del_u1y, del_u1z, del_B1x, del_B1y, del_B1z) = del_ψ
#     dρ1_dt = tf.reshape(del_ρ1[:, 0], (n, 1))
#     dρ1_dx = tf.reshape(del_ρ1[:, 1], (n, 1))
#     dP1_dt = tf.reshape(del_P1[:, 0], (n, 1))
#     dP1_dx = tf.reshape(del_P1[:, 1], (n, 1))
#     du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
#     du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
#     du1y_dt = tf.reshape(del_u1y[:, 0], (n, 1))
#     du1y_dx = tf.reshape(del_u1y[:, 1], (n, 1))
#     du1z_dt = tf.reshape(del_u1z[:, 0], (n, 1))
#     du1z_dx = tf.reshape(del_u1z[:, 1], (n, 1))
#     dB1x_dt = tf.reshape(del_B1x[:, 0], (n, 1))
#     dB1x_dx = tf.reshape(del_B1x[:, 1], (n, 1))
#     dB1y_dt = tf.reshape(del_B1y[:, 0], (n, 1))
#     dB1y_dx = tf.reshape(del_B1y[:, 1], (n, 1))
#     dB1z_dt = tf.reshape(del_B1z[:, 0], (n, 1))
#     dB1z_dx = tf.reshape(del_B1z[:, 1], (n, 1))

#     # G is a Tensor of shape (n, 1).
#     G = ρ0*du1x_dt + dP1_dx
#     return G


# # @tf.function
# def pde_u1y(X, ψ, del_ψ):
#     """Differential equation for y-velocity perturbation.

#     Evaluate the differential equation for y-velocity perturbation. This equation
#     is derived from the equation of conservation of y-momentum.

#     Parameters
#     ----------
#     X : tf.Variable, shape (n, n_dim)
#         Values of independent variables at each evaluation point.
#     ψ : list of n_var tf.Tensor, each shape (n, 1)
#         Values of dependent variables at each evaluation point.
#     del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
#         Values of gradients of dependent variables wrt independent variables at
#         each evaluation point.

#     Returns
#     -------
#     G : tf.Tensor, shape (n, 1)
#         Value of differential equation at each evaluation point.
#     """
#     n = X.shape[0]
#     t = tf.reshape(X[:, 0], (n, 1))
#     x = tf.reshape(X[:, 1], (n, 1))
#     (ρ1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = ψ
#     (del_ρ1, del_P1, del_u1x, del_u1y, del_u1z, del_B1x, del_B1y, del_B1z) = del_ψ
#     dρ1_dt = tf.reshape(del_ρ1[:, 0], (n, 1))
#     dρ1_dx = tf.reshape(del_ρ1[:, 1], (n, 1))
#     dP1_dt = tf.reshape(del_P1[:, 0], (n, 1))
#     dP1_dx = tf.reshape(del_P1[:, 1], (n, 1))
#     du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
#     du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
#     du1y_dt = tf.reshape(del_u1y[:, 0], (n, 1))
#     du1y_dx = tf.reshape(del_u1y[:, 1], (n, 1))
#     du1z_dt = tf.reshape(del_u1z[:, 0], (n, 1))
#     du1z_dx = tf.reshape(del_u1z[:, 1], (n, 1))
#     dB1x_dt = tf.reshape(del_B1x[:, 0], (n, 1))
#     dB1x_dx = tf.reshape(del_B1x[:, 1], (n, 1))
#     dB1y_dt = tf.reshape(del_B1y[:, 0], (n, 1))
#     dB1y_dx = tf.reshape(del_B1y[:, 1], (n, 1))
#     dB1z_dt = tf.reshape(del_B1z[:, 0], (n, 1))
#     dB1z_dx = tf.reshape(del_B1z[:, 1], (n, 1))

#     # G is a Tensor of shape (n, 1).
#     G = ρ0*du1y_dt - B0x*dB1y_dx/μ0
#     return G


# # @tf.function
# def pde_u1z(X, ψ, del_ψ):
#     """Differential equation for z-velocity perturbation.

#     Evaluate the differential equation for z-velocity perturbation.
#     This equation is derived from the equation of conservation of z-momentum.

#     Parameters
#     ----------
#     X : tf.Variable, shape (n, n_dim)
#         Values of independent variables at each evaluation point.
#     ψ : list of n_var tf.Tensor, each shape (n, 1)
#         Values of dependent variables at each evaluation point.
#     del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
#         Values of gradients of dependent variables wrt independent variables at
#         each evaluation point.

#     Returns
#     -------
#     G : tf.Tensor, shape (n, 1)
#         Value of differential equation at each evaluation point.
#     """
#     n = X.shape[0]
#     t = tf.reshape(X[:, 0], (n, 1))
#     x = tf.reshape(X[:, 1], (n, 1))
#     (ρ1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = ψ
#     (del_ρ1, del_P1, del_u1x, del_u1y, del_u1z, del_B1x, del_B1y, del_B1z) = del_ψ
#     dρ1_dt = tf.reshape(del_ρ1[:, 0], (n, 1))
#     dρ1_dx = tf.reshape(del_ρ1[:, 1], (n, 1))
#     dP1_dt = tf.reshape(del_P1[:, 0], (n, 1))
#     dP1_dx = tf.reshape(del_P1[:, 1], (n, 1))
#     du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
#     du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
#     du1y_dt = tf.reshape(del_u1y[:, 0], (n, 1))
#     du1y_dx = tf.reshape(del_u1y[:, 1], (n, 1))
#     du1z_dt = tf.reshape(del_u1z[:, 0], (n, 1))
#     du1z_dx = tf.reshape(del_u1z[:, 1], (n, 1))
#     dB1x_dt = tf.reshape(del_B1x[:, 0], (n, 1))
#     dB1x_dx = tf.reshape(del_B1x[:, 1], (n, 1))
#     dB1y_dt = tf.reshape(del_B1y[:, 0], (n, 1))
#     dB1y_dx = tf.reshape(del_B1y[:, 1], (n, 1))
#     dB1z_dt = tf.reshape(del_B1z[:, 0], (n, 1))
#     dB1z_dx = tf.reshape(del_B1z[:, 1], (n, 1))

#     # G is a Tensor of shape (n, 1).
#     G = ρ0*du1z_dt - B0x*dB1z_dx/μ0
#     return G


# # @tf.function
# def pde_B1x(X, ψ, del_ψ):
#     """Differential equation for x-magnetic field perturbation.

#     Evaluate the differential equation for x-magnetic field perturbation.
#     This equation is derived from the x-component of Faraday's Law.

#     Parameters
#     ----------
#     X : tf.Variable, shape (n, n_dim)
#         Values of independent variables at each evaluation point.
#     ψ : list of n_var tf.Tensor, each shape (n, 1)
#         Values of dependent variables at each evaluation point.
#     del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
#         Values of gradients of dependent variables wrt independent variables at
#         each evaluation point.

#     Returns
#     -------
#     G : tf.Tensor, shape (n, 1)
#         Value of differential equation at each evaluation point.
#     """
#     n = X.shape[0]
#     t = tf.reshape(X[:, 0], (n, 1))
#     x = tf.reshape(X[:, 1], (n, 1))
#     (ρ1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = ψ
#     (del_ρ1, del_P1, del_u1x, del_u1y, del_u1z, del_B1x, del_B1y, del_B1z) = del_ψ
#     dρ1_dt = tf.reshape(del_ρ1[:, 0], (n, 1))
#     dρ1_dx = tf.reshape(del_ρ1[:, 1], (n, 1))
#     dP1_dt = tf.reshape(del_P1[:, 0], (n, 1))
#     dP1_dx = tf.reshape(del_P1[:, 1], (n, 1))
#     du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
#     du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
#     du1y_dt = tf.reshape(del_u1y[:, 0], (n, 1))
#     du1y_dx = tf.reshape(del_u1y[:, 1], (n, 1))
#     du1z_dt = tf.reshape(del_u1z[:, 0], (n, 1))
#     du1z_dx = tf.reshape(del_u1z[:, 1], (n, 1))
#     dB1x_dt = tf.reshape(del_B1x[:, 0], (n, 1))
#     dB1x_dx = tf.reshape(del_B1x[:, 1], (n, 1))
#     dB1y_dt = tf.reshape(del_B1y[:, 0], (n, 1))
#     dB1y_dx = tf.reshape(del_B1y[:, 1], (n, 1))
#     dB1z_dt = tf.reshape(del_B1z[:, 0], (n, 1))
#     dB1z_dx = tf.reshape(del_B1z[:, 1], (n, 1))

#     # G is a Tensor of shape (n, 1).
#     G = tf.zeros((n,))
#     return G


# # @tf.function
# def pde_B1y(X, ψ, del_ψ):
#     """Differential equation for y-magnetic field perturbation.

#     Evaluate the differential equation for y-magnetic field perturbation.
#     This equation is derived from the y-component of Faraday's Law.

#     Parameters
#     ----------
#     X : tf.Variable, shape (n, n_dim)
#         Values of independent variables at each evaluation point.
#     ψ : list of n_var tf.Tensor, each shape (n, 1)
#         Values of dependent variables at each evaluation point.
#     del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
#         Values of gradients of dependent variables wrt independent variables at
#         each evaluation point.

#     Returns
#     -------
#     G : tf.Tensor, shape (n, 1)
#         Value of differential equation at each evaluation point.
#     """
#     n = X.shape[0]
#     t = tf.reshape(X[:, 0], (n, 1))
#     x = tf.reshape(X[:, 1], (n, 1))
#     (ρ1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = ψ
#     (del_ρ1, del_P1, del_u1x, del_u1y, del_u1z, del_B1x, del_B1y, del_B1z) = del_ψ
#     dρ1_dt = tf.reshape(del_ρ1[:, 0], (n, 1))
#     dρ1_dx = tf.reshape(del_ρ1[:, 1], (n, 1))
#     dP1_dt = tf.reshape(del_P1[:, 0], (n, 1))
#     dP1_dx = tf.reshape(del_P1[:, 1], (n, 1))
#     du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
#     du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
#     du1y_dt = tf.reshape(del_u1y[:, 0], (n, 1))
#     du1y_dx = tf.reshape(del_u1y[:, 1], (n, 1))
#     du1z_dt = tf.reshape(del_u1z[:, 0], (n, 1))
#     du1z_dx = tf.reshape(del_u1z[:, 1], (n, 1))
#     dB1x_dt = tf.reshape(del_B1x[:, 0], (n, 1))
#     dB1x_dx = tf.reshape(del_B1x[:, 1], (n, 1))
#     dB1y_dt = tf.reshape(del_B1y[:, 0], (n, 1))
#     dB1y_dx = tf.reshape(del_B1y[:, 1], (n, 1))
#     dB1z_dt = tf.reshape(del_B1z[:, 0], (n, 1))
#     dB1z_dx = tf.reshape(del_B1z[:, 1], (n, 1))

#     # G is a Tensor of shape (n, 1).
#     G = dB1y_dt - B0x*du1y_dx
#     return G


# # @tf.function
# def pde_B1z(X, ψ, del_ψ):
#     """Differential equation for z-magnetic field perturbation.

#     Evaluate the differential equation for z-magnetic field perturbation.
#     This equation is derived from the z-component of Faraday's Law.

#     Parameters
#     ----------
#     X : tf.Variable, shape (n, n_dim)
#         Values of independent variables at each evaluation point.
#     ψ : list of n_var tf.Tensor, each shape (n, 1)
#         Values of dependent variables at each evaluation point.
#     del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
#         Values of gradients of dependent variables wrt independent variables at
#         each evaluation point.

#     Returns
#     -------
#     G : tf.Tensor, shape (n, 1)
#         Value of differential equation at each evaluation point.
#     """
#     n = X.shape[0]
#     t = tf.reshape(X[:, 0], (n, 1))
#     x = tf.reshape(X[:, 1], (n, 1))
#     (ρ1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = ψ
#     (del_ρ1, del_P1, del_u1x, del_u1y, del_u1z, del_B1x, del_B1y, del_B1z) = del_ψ
#     dρ1_dt = tf.reshape(del_ρ1[:, 0], (n, 1))
#     dρ1_dx = tf.reshape(del_ρ1[:, 1], (n, 1))
#     dP1_dt = tf.reshape(del_P1[:, 0], (n, 1))
#     dP1_dx = tf.reshape(del_P1[:, 1], (n, 1))
#     du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
#     du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
#     du1y_dt = tf.reshape(del_u1y[:, 0], (n, 1))
#     du1y_dx = tf.reshape(del_u1y[:, 1], (n, 1))
#     du1z_dt = tf.reshape(del_u1z[:, 0], (n, 1))
#     du1z_dx = tf.reshape(del_u1z[:, 1], (n, 1))
#     dB1x_dt = tf.reshape(del_B1x[:, 0], (n, 1))
#     dB1x_dx = tf.reshape(del_B1x[:, 1], (n, 1))
#     dB1y_dt = tf.reshape(del_B1y[:, 0], (n, 1))
#     dB1y_dx = tf.reshape(del_B1y[:, 1], (n, 1))
#     dB1z_dt = tf.reshape(del_B1z[:, 0], (n, 1))
#     dB1z_dx = tf.reshape(del_B1z[:, 1], (n, 1))

#     # G is a Tensor of shape (n, 1).
#     G = dB1z_dt - B0x*du1z_dx
#     return G


# # Make a list of all of the differential equations.
# differential_equations = [
#     pde_ρ1,
#     pde_P1,
#     pde_u1x,
#     pde_u1y,
#     pde_u1z,
#     pde_B1x,
#     pde_B1y,
#     pde_B1z
# ]


# def ρ1_analytical(X: np.ndarray):
#     """Compute analytical solution for mass density perturbation.

#     Compute anaytical solution for mass density perturbation.

#     Parameters
#     ----------
#     X : np.ndarray of float, shape (n, n_dim)
#         Independent variable values for computation.

#     Returns
#     -------
#     ρ1 : np.ndarray of float, shape (n,)
#         Analytical values for mass density perturbation.
#     """
#     ρ1 = np.full((X.shape[0],), ρ1_amp)
#     return ρ1


# def P1_analytical(X: np.ndarray):
#     """Compute analytical solution for pressure perturbation.

#     Compute anaytical solution for pressure perturbation.

#     Parameters
#     ----------
#     X : np.ndarray of float, shape (n, n_dim)
#         Independent variable values for computation.

#     Returns
#     -------
#     P1 : np.ndarray of float, shape (n,)
#         Analytical values for pressure perturbation.
#     """
#     P1 = np.full((X.shape[0],), P1_amp)
#     return P1


# def u1x_analytical(X: np.ndarray):
#     """Compute analytical solution for x-velocity perturbation.

#     Compute anaytical solution for x-velocity perturbation.

#     Parameters
#     ----------
#     X : np.ndarray of float, shape (n, n_dim)
#         Independent variable values for computation.

#     Returns
#     -------
#     u1x : np.ndarray of float, shape (n,)
#         Analytical values for x-velocity perturbation.
#     """
#     u1x = np.full((X.shape[0],), u1x_amp)
#     return u1x


# def u1y_analytical(X: np.ndarray):
#     """Compute analytical solution for y-velocity perturbation.

#     Compute anaytical solution for y-velocity perturbation.

#     Parameters
#     ----------
#     X : np.ndarray of float, shape (n, n_dim)
#         Independent variable values for computation.

#     Returns
#     -------
#     uy : np.ndarray of float, shape (n,)
#         Analytical values for y-velocity perturbation.
#     """
#     t = X[:, 0]
#     x = X[:, 1]
#     u1y = u1y_amp*np.sin(kx*x - ω*t)
#     return u1y


# def u1z_analytical(X: np.ndarray):
#     """Compute analytical solution for z-velocity perturbation.

#     Compute anaytical solution for z-velocity perturbation.

#     Parameters
#     ----------
#     X : np.ndarray of float, shape (n, n_dim)
#         Independent variable values for computation.

#     Returns
#     -------
#     u1z : np.ndarray of float, shape (n,)
#         Analytical values for z-velocity perturbation.
#     """
#     u1z = np.full((X.shape[0],), u1z_amp)
#     return u1z


# def B1x_analytical(X: np.ndarray):
#     """Compute analytical solution for x-magnetic field perturbation.

#     Compute anaytical solution for x-magnetic field perturbation.

#     Parameters
#     ----------
#     X : np.ndarray of float, shape (n, n_dim)
#         Independent variable values for computation.

#     Returns
#     -------
#     B1x : np.ndarray of float, shape (n,)
#         Analytical values for x-magnetic field perturbation.
#     """
#     t = X[:, 0]
#     x = X[:, 1]
#     B1x = np.zeros((len(t),))
#     return B1x


# def B1y_analytical(X: np.ndarray):
#     """Compute analytical solution for y-magnetic field perturbation.

#     Compute anaytical solution for y-magnetic field perturbation.

#     Parameters
#     ----------
#     X : np.ndarray of float, shape (n, n_dim)
#         Independent variable values for computation.

#     Returns
#     -------
#     B1y : np.ndarray of float, shape (n,)
#         Analytical values for y-magnetic field perturbation.
#     """
#     t = X[:, 0]
#     x = X[:, 1]
#     B1y = B1y_amp*np.sin(kx*x - ω*t + np.pi)
#     return B1y


# def B1z_analytical(X: np.ndarray):
#     """Compute analytical solution for z-magnetic field perturbation.

#     Compute anaytical solution for z-magnetic field perturbation.

#     Parameters
#     ----------
#     X : np.ndarray of float, shape (n, n_dim)
#         Independent variable values for computation.

#     Returns
#     -------
#     B1z : np.ndarray of float, shape (n,)
#         Analytical values for z-magnetic field perturbation.
#     """
#     B1z = np.full((X.shape[0],), B1z_amp)
#     return B1z


# # Gather all of the analytical solutions into a list.
# ψ_analytical = [
#     ρ1_analytical,
#     P1_analytical,
#     u1x_analytical,
#     u1y_analytical,
#     u1z_analytical,
#     B1x_analytical,
#     B1y_analytical,
#     B1z_analytical,
# ]


if __name__ == "__main__":
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_index = {independent_variable_index}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")

    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_index = {dependent_variable_index}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    # print(f"μ0 = {μ0}")
    # print(f"m = {m}")
    # print(f"ɣ = {ɣ}")

    # print(f"n0 = {n0}")
    # print(f"P0 = {P0}")
    # print(f"u0x = {u0x}")
    # print(f"u0y = {u0y}")
    # print(f"u0z = {u0z}")
    # print(f"B0x = {B0x}")
    # print(f"B0y = {B0y}")
    # print(f"B0z = {B0z}")
