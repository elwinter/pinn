# Import standard modules.
from importlib import import_module
import os
import sys

# Import supplemental modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Import project modules.
import pinn.standard_plots as psp

# Specify the run ID (aka problem name).
runid = "lagaris01"

# Add the subdirectory for the run results to the module search path.
run_path = os.path.join(".", runid)
sys.path.append(run_path)

# Import the problem definition from the run results directory.
p = import_module(runid)

# Read the run hyperparameters from the run results directory.
import hyperparameters as hp

# Load all data.

X_train = np.loadtxt(os.path.join(runid, "X_train.dat"))
x_train = X_train  # 1-D, so p.ix not needed

# Load the data locations and values (includes initial conditions).
XY_data = np.loadtxt(os.path.join(runid, "XY_data.dat"))

# Extract the initial conditions (everything after the coordinate values on each row).
ic = XY_data[p.n_dim:]

# Load the model-predicted values.
ψ = []
delψ = []
for var_name in p.dependent_variable_names:
    ψ.append(np.loadtxt(os.path.join(runid, "%s_train.dat" % var_name)))
    delψ.append(np.loadtxt(os.path.join(runid, "del_%s_train.dat" % var_name)))

# Load the loss function histories.
losses_model = np.loadtxt(os.path.join(runid, "losses_model.dat"))
losses_model_res = np.loadtxt(os.path.join(runid, "losses_model_res.dat"))
losses_model_data = np.loadtxt(os.path.join(runid, "losses_model_data.dat"))
losses = np.loadtxt(os.path.join(runid, "losses.dat"))
losses_res = np.loadtxt(os.path.join(runid, "losses_res.dat"))
losses_data = np.loadtxt(os.path.join(runid, "losses_data.dat"))

# Compute the limits of the training domain.
x_min = x_train[0]
x_max = x_train[-1]

# Extract the unique training point values (a grid is assumed).
x_train_vals = np.unique(x_train)
n_x_train_vals = len(x_train_vals)

# Plotting options

# Specify the size (width, height) (in inches) for individual subplots.
SUBPLOT_WIDTH = 5.0
SUBPLOT_HEIGHT = 5.0

# Compute the coordinate plot tick locations and labels.
XY_N_X_TICKS = 5
XY_x_tick_pos = np.linspace(x_min, x_max, XY_N_X_TICKS)
XY_x_tick_labels = ["%.1f" % x for x in XY_x_tick_pos]

# Create figures in a memory buffer.
mpl.use("Agg")

# Plot the total loss function history.
total_loss_figsize = (SUBPLOT_WIDTH*2, SUBPLOT_HEIGHT)
plt.figure(figsize=total_loss_figsize)
psp.plot_loss_functions(
    [losses_res, losses_data, losses],
    ["$L_{res}$", "$L_{data}$", "$L$"],
    title="Total loss function history for %s" % runid
)
plt.savefig("loss.png")

# Extract the coordinates of the training points at the initial time.
n_start = n_x_train_vals
x0 = XY_data[0]
y0 = XY_data[-1]

# Extract the trained solution.
y_train = ψ[0]

# Compute the analytical solution at the training points.
y_analytical = p.Ψ_analytical(x_train)

# Compute the error and RMS error in the trained solution.
y_err = y_train - y_analytical
y_rms_err = np.sqrt(np.sum(y_err**2)/len(y_err))

# Plot the actual, predicted, and absolute error in the solution.
plt.clf()
ax1 = plt.gca()
ax1.plot(x_train, y_train, label="$\psi_t$")
ax1.plot(x_train, y_analytical, label="$\psi_a$")
ax1.plot(x_train, y_err, label="$\psi_{err}$")
ax1.legend()
ax1.set_xlabel(p.independent_variable_labels[0])
ax1.set_ylabel(p.dependent_variable_labels[0])
ax1.set_title(f"Trained and analytical solutions for {runid}\n"
              f"RMS error = {y_rms_err:.2e}")
plt.savefig("trained_actual_error.png")

# Extract the trained derivative.
dy_dx_train = delψ[0]

# Compute the analytical solution at the training points.
dy_dx_analytical = p.dΨ_dx_analytical(x_train)

# Compute the error and RMS error in the trained derivative.
dy_dx_err = dy_dx_train - dy_dx_analytical
dy_dx_rms_err = np.sqrt(np.sum(dy_dx_err**2)/len(dy_dx_err))

# Plot the actual, predicted, and absolute error in the derivative.
plt.clf()
ax1 = plt.gca()
ax1.plot(x_train, dy_dx_train, label="$d\psi_t/dx$")
ax1.plot(x_train, dy_dx_analytical, label="$d\psi_a/dx$")
ax1.plot(x_train, y_err, label="$d\psi/dx (err)$")
ax1.legend()
ax1.set_xlabel(p.independent_variable_labels[0])
ax1.set_ylabel("$d\psi/dx$")
ax1.set_title(f"Trained and analytical derivative for {runid}\n"
              f"RMS error = {dy_dx_rms_err:.2e}")
plt.savefig("trained_actual_derivative_error.png")

