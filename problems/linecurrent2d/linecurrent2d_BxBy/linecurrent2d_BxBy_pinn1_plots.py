#!/usr/bin/env python

"""Create plots for pinn1 results for linecurrent2d_BxBy problem.

Create plots for pinn1 results for linecurrent2d_BxBy problem.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

# Import standard modules.
import argparse
from importlib import import_module
import os
import sys

# Import supplemental modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import project modules.
import pinn.common


# Program constants

# Program description
DESCRIPTION = "Create plots for pinn1 results for linecurrent2d_BxBy problem."

# Name of directory to hold output plots
OUTPUT_DIR = "pinn1_plots"

# Name of problem
PROBLEM_NAME = "linecurrent2d_BxBy"

# Plot limits for dependent variables.
ylim = {}
ylim["L"] = [1e-12, 10]
ylim["Bx"] = [-1.0, 1.0]
ylim["By"] = [-1.0, 1.0]


def create_command_line_argument_parser():
    """Create the command-line argument parser.

    Create the command-line argument parser.

    Parameters
    ----------
    None

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser for command-line arguments.
    """
    parser = argparse.ArgumentParser(DESCRIPTION)
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "results_path",
        help="Path to directory containing results to plot."
    )
    return parser


def main():
    """Main program."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    debug = args.debug
    verbose = args.verbose
    results_path = args.results_path
    if debug:
        print(f"args = {args}", flush=True)

    # Add the run results directory to the module search path.
    sys.path.append(results_path)

    # Import the problem definition from the run results directory.
    p = import_module(PROBLEM_NAME)

    # Compute the path to the output directory. Then create it if needed.
    output_path = OUTPUT_DIR
    os.mkdir(output_path)

    # Create the plots in a memory buffer.
    mpl.use("Agg")

    # -------------------------------------------------------------------------

# # Specify the run ID (aka problem name).
# runid = "linecurrent_BxBy"

# # Add the subdirectory for the run results to the module search path.
# run_path = os.path.join(".", runid)
# sys.path.append(run_path)

# # Import the problem definition from the run results directory.
# p = import_module(runid)

# # Read the run hyperparameters from the run results directory.
# import hyperparameters as hp

# # Load the training point coordinates.
# X_train = np.loadtxt(os.path.join(runid, "X_train.dat"))
# t_train = X_train[:, p.it]
# x_train = X_train[:, p.ix]
# y_train = X_train[:, p.iy]

# # Load the data locations and values (includes initial conditions).
# XY_data = np.loadtxt(os.path.join(runid, "XY_data.dat"))

# # Extract the initial conditions (everything after the coordinate values on each row).
# ic = XY_data[p.n_dim:]

# # Load the model-predicted values.
# ψ = []
# delψ = []
# for var_name in p.dependent_variable_names:
#     ψ.append(np.loadtxt(os.path.join(runid, "%s_train.dat" % var_name)))
#     delψ.append(np.loadtxt(os.path.join(runid, "del_%s_train.dat" % var_name)))

# # Load the loss function histories.
# losses_model = np.loadtxt(os.path.join(runid, "losses_model.dat"))
# losses_model_res = np.loadtxt(os.path.join(runid, "losses_model_res.dat"))
# losses_model_data = np.loadtxt(os.path.join(runid, "losses_model_data.dat"))
# losses = np.loadtxt(os.path.join(runid, "losses.dat"))
# losses_res = np.loadtxt(os.path.join(runid, "losses_res.dat"))
# losses_data = np.loadtxt(os.path.join(runid, "losses_data.dat"))

# # Compute the limits of the training domain.
# t_min = t_train[0]
# t_max = t_train[-1]
# x_min = x_train[0]
# x_max = x_train[-1]
# y_min = y_train[0]
# y_max = y_train[-1]

# # Extract the unique training point values (a grid is assumed).
# t_train_vals = np.unique(t_train)
# x_train_vals = np.unique(x_train)
# y_train_vals = np.unique(y_train)
# n_t_train_vals = len(t_train_vals)
# n_x_train_vals = len(x_train_vals)
# n_y_train_vals = len(y_train_vals)

# # Plotting options

# # Specify the size (width, height) (in inches) for individual subplots.
# SUBPLOT_WIDTH = 5.0
# SUBPLOT_HEIGHT = 5.0

# # Compute the coordinate plot tick locations and labels.
# XY_N_X_TICKS = 5
# XY_x_tick_pos = np.linspace(x_min, x_max, XY_N_X_TICKS)
# XY_x_tick_labels = ["%.1f" % x for x in XY_x_tick_pos]
# XY_N_Y_TICKS = 5
# XY_y_tick_pos = np.linspace(y_min, y_max, XY_N_Y_TICKS)
# XY_y_tick_labels = ["%.1f" % y for y in XY_y_tick_pos]

# # Compute the heat map tick locations and labels.
# HEATMAP_N_X_TICKS = 5
# heatmap_x_tick_pos = np.linspace(0, n_x_train_vals - 1, HEATMAP_N_X_TICKS)
# heatmap_x_tick_labels = ["%.1f" % (x_min + x/(n_x_train_vals - 1)*(x_max - x_min)) for x in heatmap_x_tick_pos]
# HEATMAP_N_Y_TICKS = 5
# heatmap_y_tick_pos = np.linspace(0, n_y_train_vals - 1, HEATMAP_N_Y_TICKS)
# heatmap_y_tick_labels = ["%.1f" % (y_min + y/(n_y_train_vals - 1)*(y_max - y_min)) for y in heatmap_y_tick_pos]
# heatmap_y_tick_labels = list(reversed(heatmap_y_tick_labels))

# # Create figures in a memory buffer.
# mpl.use("Agg")

# # Plot the loss history for each model.
# fig = psp.plot_model_loss_functions(
#     losses_model_res, losses_model_data, losses_model,
#     p.dependent_variable_labels
# )
# plt.savefig("model_losses.png")

# # Plot the total loss function history.
# total_loss_figsize = (SUBPLOT_WIDTH*2, SUBPLOT_HEIGHT)
# plt.figure(figsize=total_loss_figsize)
# psp.plot_loss_functions(
#     [losses_res, losses_data, losses],
#     ["$L_{res}$", "$L_{data}$", "$L$"],
#     title="Total loss function history for %s" % runid
# )
# plt.savefig("loss.png")

# # Extract the coordinates of the training points at the initial time.
# n_start = n_x_train_vals*n_y_train_vals
# t0 = XY_data[:, p.it]
# x0 = XY_data[:, p.ix]
# y0 = XY_data[:, p.iy]

# # Plot the actual and predicted initial magnetic field vectors.
# B0x_act = p.Bx_analytical(t0, x0, y0)
# B0y_act = p.By_analytical(t0, x0, y0)
# B0x_pred = ψ[p.iBx][:n_start]
# B0y_pred = ψ[p.iBy][:n_start]

# # Create the figure.
# fig = psp.plot_actual_predicted_B(
#     x0, y0, B0x_act, B0y_act, B0x_pred, B0y_pred,
#     title="Initial magnetic field",
#     x_tick_pos=XY_x_tick_pos, x_tick_labels=XY_x_tick_labels,
#     y_tick_pos=XY_y_tick_pos, y_tick_labels=XY_y_tick_labels,
# )
# plt.savefig("B0xB0y.png")

# # Plot the actual, predicted, and absolute error in initial magnetic field magnitudes.
# B0_act = np.sqrt(B0x_act**2 + B0y_act**2)
# B0_pred = np.sqrt(B0x_pred**2 + B0y_pred**2)
# B0_err = B0_pred - B0_act

# # To get the proper orientation, reshape, transpose, flip.
# B0_act_plot = np.flip(B0_act.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# B0_pred_plot = np.flip(B0_pred.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# B0_err_plot = np.flip(B0_err.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)

# # Create the plot.
# B_MIN = 1e-4
# B_MAX = 1e-2
# B_ERR_MIN = -1e-3
# B_ERR_MAX = 1e-3
# fig = psp.plot_log_actual_predicted_error(
#     x0, y0, B0_act_plot, B0_pred_plot, B0_err_plot,
#     vmin=B_MIN, vmax=B_MAX, err_vmin=B_ERR_MIN, err_vmax=B_ERR_MAX,
#     title="Initial magnetic field magnitude",
#     x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
# )
# plt.savefig("B0.png")

# # Plot the actual, predicted, and absolute error in initial magnetic field divergence.
# dB0x_dx_act = p.dBx_dx_analytical(t0, x0, y0)
# dB0y_dy_act = p.dBy_dy_analytical(t0, x0, y0)
# divB0_act = dB0x_dx_act + dB0y_dy_act
# dB0x_dx_pred = delψ[p.iBx][:n_start, p.ix]
# dB0y_dy_pred = delψ[p.iBy][:n_start, p.iy]
# divB0_pred = dB0x_dx_pred + dB0y_dy_pred
# divB0_err = divB0_pred - divB0_act

# # To get the proper orientation, reshape, transpose, flip.
# divB0_act_plot = np.flip(divB0_act.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# divB0_pred_plot = np.flip(divB0_pred.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# divB0_err_plot = np.flip(divB0_err.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)

# # Create the figure.
# DIVB_MIN = -1e-1
# DIVB_MAX = 1e-1
# DIVB_ERR_MIN = -1e-1
# DIVB_ERR_MAX = 1e-1
# fig = psp.plot_actual_predicted_error(
#     x0, y0, divB0_act_plot, divB0_pred_plot, divB0_err_plot,
#     vmin=DIVB_MIN, vmax=DIVB_MAX, err_vmin=DIVB_ERR_MIN, err_vmax=DIVB_ERR_MAX,
#     title="Initial magnetic divergence",
#     x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
# )
# plt.savefig("divB0.png")

# # Plot the actual, predicted, and absolute errors in initial Bx.
# B0x_err = B0x_pred - B0x_act

# # To get the proper orientation, reshape, transpose, flip.
# B0x_act_plot = np.flip(B0x_act.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# B0x_pred_plot = np.flip(B0x_pred.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# B0x_err_plot = np.flip(B0x_err.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)

# # Create the figure.
# BX_MIN = -5e-3
# BX_MAX = 5e-3
# BX_ERR_MIN = -1e-3
# BX_ERR_MAX = 1e-3
# fig = psp.plot_actual_predicted_error(
#     x0, y0, B0x_act_plot, B0x_pred_plot, B0x_err_plot,
#     title="Initial %s" % p.dependent_variable_labels[p.iBx],
#     vmin=BX_MIN, vmax=BX_MAX, err_vmin=BX_ERR_MIN, err_vmax=BX_ERR_MAX,
#     x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
# )
# plt.savefig("B0x.png")

# # Plot the actual, predicted, and absolute errors in initial By.
# B0y_err = B0y_pred - B0y_act

# # To get the proper orientation, reshape, transpose, flip.
# B0y_act_plot = np.flip(B0y_act.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# B0y_pred_plot = np.flip(B0y_pred.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# B0y_err_plot = np.flip(B0y_err.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)

# # Create the figure.
# BY_MIN = -5e-3
# BY_MAX = 5e-3
# BY_ERR_MIN = -1e-3
# BY_ERR_MAX = 1e-3
# fig = psp.plot_actual_predicted_error(
#     x0, y0, B0y_act_plot, B0y_pred_plot, B0y_err_plot,
#     title="Initial %s" % p.dependent_variable_labels[p.iBy],
#     vmin=BY_MIN, vmax=BY_MAX, err_vmin=BY_ERR_MIN, err_vmax=BY_ERR_MAX,
#     x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
# )
# plt.savefig("B0y.png")

# # Plot the actual, predicted, and absolute errors in initial dBx/dx.
# dB0x_dx_err = dB0x_dx_pred - dB0x_dx_act

# # To get the proper orientation, reshape, transpose, flip.
# dB0x_dx_act_plot = np.flip(dB0x_dx_act.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# dB0x_dx_pred_plot = np.flip(dB0x_dx_pred.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# dB0x_dx_err_plot = np.flip(dB0x_dx_err.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)

# # Create the figure.
# DBX_DX_MIN = -0.2
# DBX_DX_MAX = 0.2
# DBX_DX_ERR_MIN = -0.2
# DBX_DX_ERR_MAX = 0.2
# fig = psp.plot_actual_predicted_error(
#     x0, y0, dB0x_dx_act_plot, dB0x_dx_pred_plot, dB0x_dx_err_plot,
#     title="Initial d%s/d%s" % (p.dependent_variable_labels[p.iBx], p.independent_variable_labels[p.ix]),
#     vmin=DBX_DX_MIN, vmax=DBX_DX_MAX, err_vmin=DBX_DX_ERR_MIN, err_vmax=DBX_DX_ERR_MAX,
#     x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
# )
# plt.savefig("dB0x_dx.png")

# # Plot the actual, predicted, and absolute errors in initial dBy/dy.
# dB0y_dy_err = dB0y_dy_pred - dB0y_dy_act

# # To get the proper orientation, reshape, transpose, flip.
# dB0y_dy_act_plot = np.flip(dB0y_dy_act.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# dB0y_dy_pred_plot = np.flip(dB0y_dy_pred.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# dB0y_dy_err_plot = np.flip(dB0y_dy_err.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)

# # Create the figure.
# DBY_DY_MIN = -0.2
# DBY_DY_MAX = 0.2
# DBY_DY_ERR_MIN = -0.2
# DBY_DY_ERR_MAX = 0.2
# fig = psp.plot_actual_predicted_error(
#     x0, y0, dB0y_dy_act_plot, dB0y_dy_pred_plot, dB0y_dy_err_plot,
#     title="Initial d%s/d%s" % (p.dependent_variable_labels[p.iBy], p.independent_variable_labels[p.iy]),
#     vmin=DBY_DY_MIN, vmax=DBY_DY_MAX, err_vmin=DBY_DY_ERR_MIN, err_vmax=DBY_DY_ERR_MAX,
#     x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
# )
# plt.savefig("dB0y_dy.png")

# # Extract the coordinates of the training points at the final time.
# n_end = n_x_train_vals*n_y_train_vals
# t1 = t_train[-n_end:]
# x1 = x_train[-n_end:]
# y1 = y_train[-n_end:]

# # Plot the actual and predicted final magnetic field vectors.
# B1x_act = p.Bx_analytical(t1, x1, y1)
# B1y_act = p.By_analytical(t1, x1, y1)
# B1x_pred = ψ[p.iBx][-n_end:]
# B1y_pred = ψ[p.iBy][-n_end:]

# # Create the figure.
# fig = psp.plot_actual_predicted_B(
#     x1, y1, B1x_act, B1y_act, B1x_pred, B1y_pred,
#     title="Final magnetic field",
#     x_tick_pos=XY_x_tick_pos, x_tick_labels=XY_x_tick_labels,
#     y_tick_pos=XY_y_tick_pos, y_tick_labels=XY_y_tick_labels,
# )
# plt.savefig("B1xB1y.png")

# # Plot the actual, predicted, and absolute error in final magnetic field magnitudes.
# B1_act = np.sqrt(B1x_act**2 + B1y_act**2)
# B1_pred = np.sqrt(B1x_pred**2 + B1y_pred**2)
# B1_err = B1_pred - B1_act

# # To get the proper orientation, reshape, transpose, flip.
# B1_act_plot = np.flip(B1_act.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# B1_pred_plot = np.flip(B1_pred.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# B1_err_plot = np.flip(B1_err.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)

# # Create the plot.
# fig = psp.plot_log_actual_predicted_error(
#     x1, y1, B1_act_plot, B1_pred_plot, B1_err_plot,
#     vmin=B_MIN, vmax=B_MAX, err_vmin=B_ERR_MIN, err_vmax=B_ERR_MAX,
#     title="Final magnetic field magnitude",
#     x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
# )
# plt.savefig("B1.png")

# # Plot the actual, predicted, and absolute error in final magnetic field divergence.
# dB1x_dx_act = p.dBx_dx_analytical(t1, x1, y1)
# dB1y_dy_act = p.dBy_dy_analytical(t1, x1, y1)
# divB1_act = dB1x_dx_act + dB1y_dy_act
# dB1x_dx_pred = delψ[p.iBx][-n_end:, p.ix]
# dB1y_dy_pred = delψ[p.iBy][-n_end:, p.iy]
# divB1_pred = dB1x_dx_pred + dB1y_dy_pred
# divB1_err = divB1_pred - divB1_act

# # To get the proper orientation, reshape, transpose, flip.
# divB1_act_plot = np.flip(divB1_act.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# divB1_pred_plot = np.flip(divB1_pred.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# divB1_err_plot = np.flip(divB1_err.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)

# # Create the figure.
# fig = psp.plot_actual_predicted_error(
#     x1, y1, divB1_act_plot, divB1_pred_plot, divB1_err_plot,
#     vmin=DIVB_MIN, vmax=DIVB_MAX, err_vmin=DIVB_ERR_MIN, err_vmax=DIVB_ERR_MAX,
#     title="Final magnetic divergence",
#     x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
# )
# plt.savefig("divB1.png")

# # Plot the actual, predicted, and absolute errors in final Bx.
# B1x_err = B1x_pred - B1x_act

# # To get the proper orientation, reshape, transpose, flip.
# B1x_act_plot = np.flip(B1x_act.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# B1x_pred_plot = np.flip(B1x_pred.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# B1x_err_plot = np.flip(B1x_err.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)

# # Create the figure.
# fig = psp.plot_actual_predicted_error(
#     x1, y1, B1x_act_plot, B1x_pred_plot, B1x_err_plot,
#     title="Final %s" % p.dependent_variable_labels[p.iBx],
#     vmin=BX_MIN, vmax=BX_MAX, err_vmin=BX_ERR_MIN, err_vmax=BX_ERR_MAX,
#     x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
# )
# plt.savefig("B1x.png")

# # Plot the actual, predicted, and absolute errors in final By.
# B1y_err = B1y_pred - B1y_act

# # To get the proper orientation, reshape, transpose, flip.
# B1y_act_plot = np.flip(B1y_act.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# B1y_pred_plot = np.flip(B1y_pred.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# B1y_err_plot = np.flip(B1y_err.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)

# # Create the figure.
# fig = psp.plot_actual_predicted_error(
#     x1, y1, B1y_act_plot, B1y_pred_plot, B1y_err_plot,
#     title="Final %s" % p.dependent_variable_labels[p.iBy],
#     vmin=BY_MIN, vmax=BY_MAX, err_vmin=BY_ERR_MIN, err_vmax=BY_ERR_MAX,
#     x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
# )
# plt.savefig("B1y.png")

# # Plot the actual, predicted, and absolute errors in final dBx/dx.
# dB1x_dx_err = dB1x_dx_pred - dB1x_dx_act

# # To get the proper orientation, reshape, transpose, flip.
# dB1x_dx_act_plot = np.flip(dB1x_dx_act.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# dB1x_dx_pred_plot = np.flip(dB1x_dx_pred.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# dB1x_dx_err_plot = np.flip(dB1x_dx_err.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)

# fig = psp.plot_actual_predicted_error(
#     x1, y1, dB1x_dx_act_plot, dB1x_dx_pred_plot, dB1x_dx_err_plot,
#     title="Final d%s/d%s" % (p.dependent_variable_labels[p.iBx], p.independent_variable_labels[p.ix]),
#     vmin=DBX_DX_MIN, vmax=DBX_DX_MAX, err_vmin=DBX_DX_ERR_MIN, err_vmax=DBX_DX_ERR_MAX,
#     x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
# )
# plt.savefig("dB1x_dx.png")

# # Plot the actual, predicted, and absolute errors in final dBy/dy.
# dB1y_dy_err = dB1y_dy_pred - dB1y_dy_act

# # To get the proper orientation, reshape, transpose, flip.
# dB1y_dy_act_plot = np.flip(dB1y_dy_act.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# dB1y_dy_pred_plot = np.flip(dB1y_dy_pred.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)
# dB1y_dy_err_plot = np.flip(dB1y_dy_err.reshape(n_x_train_vals, n_y_train_vals).T, axis=0)

# # Create the figure.
# fig = psp.plot_actual_predicted_error(
#     x1, y1, dB1y_dy_act_plot, dB1y_dy_pred_plot, dB1y_dy_err_plot,
#     title="Final d%s/d%s" % (p.dependent_variable_labels[p.iBy], p.independent_variable_labels[p.iy]),
#     vmin=DBY_DY_MIN, vmax=DBY_DY_MAX, err_vmin=DBY_DY_ERR_MIN, err_vmax=DBY_DY_ERR_MAX,
#     x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
# )
# plt.savefig("dB1y_dy.png")


if __name__ == "__main__":
    """Begin main program."""
    main()
