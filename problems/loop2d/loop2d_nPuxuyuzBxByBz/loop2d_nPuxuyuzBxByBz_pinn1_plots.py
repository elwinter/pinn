#!/usr/bin/env python

"""Create plots for pinn1 results for loop2d_nPuxuyuzBxByBz problem.

Create plots for pinn1 results for loop2d_nPuxuyuzBxByBz problem.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

# Import standard modules.
import argparse
import copy
import importlib
import os
import shutil
import sys

# Import supplemental modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import project modules.
import pinn.common
# import pinn.standard_plots


# Program constants

# Program description
DESCRIPTION = (
    "Create plots for pinn1 results for loop2d_nPuxuyuzBxByBz problem."
)

# Default values for command-line arguments.
DEFAULT_ARGUMENTS = {
    "clobber": False,
    "debug": False,
    "verbose": False,
}

# Name of problem
PROBLEM_NAME = "loop2d_nPuxuyuzBxByBz"

# Name of directory to hold output plots
OUTPUT_DIR = "pinn1_plots"

# # Plot limits for dependent variables.
# ylim = {}
# ylim["L"] = [1e-12, 10]
# ylim["Bx"] = [-1.0, 1.0]
# ylim["By"] = [-1.0, 1.0]


def create_command_line_parser():
    """Create the command-line parser.

    Create the command-line parser.

    Parameters
    ----------
    None

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser for command-line arguments.

    Raises
    ------
    None
    """
    parser = argparse.ArgumentParser(DESCRIPTION)
    parser.add_argument(
        "--clobber",
        default=DEFAULT_ARGUMENTS["clobber"],
        action="store_true",
        help="Overwrite existing plots (default: %(default)s)."
    )
    parser.add_argument(
        "--debug", "-d",
        default=DEFAULT_ARGUMENTS["debug"],
        action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--verbose", "-v",
        default=DEFAULT_ARGUMENTS["verbose"],
        action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "results_path",
        help="Path to directory containing results to plot."
    )
    return parser


def create_loss_plot(L_res: np.ndarray, L_dat: np.ndarray,
                     L: np.ndarray) -> mpl.pyplot.Figure:
    """Create a plot of residual, model, and weighted loss.

    Create a plot of residual, model, and weighted loss.

    Parameters
    ----------
    L_res : np.ndarray, shape (n_epochs,)
        Residual loss values
    L_dat : np.ndarray, shape (n_epochs,)
        Data loss values
    L : np.ndarray, shape (n_epochs,)
        Weighted loss values

    Returns
    -------
    fig : mpl.pyplot.Figure
        Figure object for current plot

    Raises
    ------
    None
    """
    # Create the figure.
    fig = plt.figure()

    # Plot the residual, data, and weighted losses.
    plt.semilogy(L_res, label="$L_{res}$")
    plt.semilogy(L_dat, label="$L_{dat}$")
    plt.semilogy(L, label="$L$")

    # Decorate the plot.
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Residual, data, and weighted loss")
    plt.grid()

    # Return the figure.
    return fig


def create_PAE_plot(X: np.ndarray, Y: np.ndarray,
                    P: np.ndarray, A: np.ndarray,
                    E: np.ndarray) -> mpl.pyplot.Figure:
    """Create a plot of predicted, ana;ytical, and error values.

    Create a plot of predicted, ana;ytical, and error values.

    Parameters
    ----------
    X : np.ndarray, shape (ny, nx)
        X values
    Y : np.ndarray, shape (ny, nx)
        Y values
    P : np.ndarray, shape (ny, nx)
        Predicted values
    A : np.ndarray, shape (ny, nx)
        Analytical values
    E : np.ndarray, shape (ny, nx)
        Error values

    Returns
    -------
    fig : mpl.pyplot.Figure
        Figure object for current plot

    Raises
    ------
    None
    """
    # Create the figure.
    fig, axs = plt.subplots(
        nrows=1, ncols=3, sharey=True,
        figsize=[18.0, 6.0]
        )

    # Predicted
    ax = axs[0]
    ax.pcolormesh(X, Y, P)
    ax.set_aspect("equal")
    ax.set_title("Predicted")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # Analytical
    ax = axs[1]
    ax.pcolormesh(X, Y, A)
    ax.set_aspect("equal")
    ax.set_title("Analytical")
    ax.set_xlabel("x")
    ax.grid(True)

    # Error
    ax = axs[2]
    ax.pcolormesh(X, Y, E)
    ax.set_aspect("equal")
    ax.set_title("Error")
    ax.set_xlabel("x")
    ax.grid(True)

    # Decorate the figure.
    fig.suptitle("Predicted, analytical, and error")

    # Return the figure.
    return fig


def pinn1_plots(**kwargs) -> int:
    """Create pinn1 plots for the loop2d_nPuxuyuzBxByBz problem.

    Create pinn1 plots for the loop2d_nPuxuyuzBxByBz problem.

    Parameters
    ----------
    kwargs : dict
        Dictionary of keyword arguments.

    Returns
    -------
    int 0 on success, otherwise Exception is raised by called code.

    Raises
    ------
    None
    """
    # Set defaults for command-line options, then update with values passed
    # from the caller.
    args = copy.deepcopy(DEFAULT_ARGUMENTS)
    args.update(kwargs)

    # Local convenience variables.
    debug = args["debug"]
    verbose = args["verbose"]
    results_path = args["results_path"]
    if debug:
        print(f"debug = {debug}")
        print(f"verbose = {verbose}")
        print(f"results_path = {results_path}")

    # ------------------------------------------------------------------------

    # Add the run results directory to the module search path.
    sys.path.append(results_path)

    # Import the problem definition from the run results directory.
    p = importlib.import_module(PROBLEM_NAME)

    # Compute the path to the output directory to hold the plots. Then create
    # it if needed. Exception will be raised if directory exists and clobber
    # is not set.
    output_path = OUTPUT_DIR
    if os.path.isdir(output_path):
        if args["clobber"]:
            if verbose:
                print(f"Deleting existing output directory {output_path}.")
            shutil.rmtree(output_path)
    os.mkdir(output_path)

    # ------------------------------------------------------------------------

    # Create the plots in a memory buffer.
    mpl.use("Agg")

    # ------------------------------------------------------------------------

    # Plot the total residual, data, and weighted loss histories.

    # Load the data.
    if verbose:
        print("Loading aggregate loss data.")
    path = os.path.join(results_path, "L_res.dat")
    L_res = np.loadtxt(path)
    path = os.path.join(results_path, "L_data.dat")
    L_dat = np.loadtxt(path)
    path = os.path.join(results_path, "L.dat")
    L = np.loadtxt(path)

    # Create the plot.
    if verbose:
        print("Creating aggregate loss plot.")
    fig = create_loss_plot(L_res, L_dat, L)
    ax = fig.get_axes()[0]
    ax.set_title("Aggregate residual, data, and weighted loss")

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "L.png")
    if verbose:
        print(f"Saving {path}.")
    fig.savefig(path)

    # ------------------------------------------------------------------------

    # Plot the per-model residual, data, and weighted loss histories.

    # Plot for each model.
    for iv in range(p.n_var):
        variable_name = p.dependent_variable_names[iv]
        if verbose:
            print(f"Creating loss plot for {variable_name}.")

        # Load the data.
        path = os.path.join(results_path, f"L_res_{variable_name}.dat")
        L_res = np.loadtxt(path)
        path = os.path.join(results_path, f"L_data_{variable_name}.dat")
        L_dat = np.loadtxt(path)
        path = os.path.join(results_path, f"L_{variable_name}.dat")
        L = np.loadtxt(path)

        # Create the plot.
        variable_label = p.dependent_variable_labels[iv]
        fig = create_loss_plot(L_res, L_dat, L)
        ax = fig.get_axes()[0]
        ax.set_title(f"{variable_label} residual, data, and weighted loss")

        # Save the plot to a PNG file.
        path = os.path.join(output_path, f"L_{variable_name}.png")
        fig.savefig(path)
        plt.close(fig)

    # ------------------------------------------------------------------------

    # Load the training points and description.
    path = os.path.join(results_path, "X_train.dat")
    X_train = np.loadtxt(path)
    with open(path, "r") as f:
        line = f.readline()  # Skip 1st line - contains "# GRID"
        line = f.readline()  # Grid description on this line
        line = line[2:]
        fields = line.split(" ")
        tmin = float(fields[0])
        tmax = float(fields[1])
        nt = int(fields[2])
        xmin = float(fields[3])
        xmax = float(fields[4])
        nx = int(fields[5])
        ymin = float(fields[6])
        ymax = float(fields[7])
        ny = int(fields[8])
    if debug:
        print(f"(tmin, tmax, nt) = ({tmin}, {tmax}, {nt})")
        print(f"(xmin, xmax, nx) = ({xmin}, {xmax}, {nx})")
        print(f"(ymin, ymax, ny) = ({ymin}, {ymax}, {ny})")

    # Find the epoch of the last trained model.
    last_epoch = pinn.common.find_last_epoch(results_path)

    # Load the trained model for each variable.
    models = []
    for variable_name in p.dependent_variable_names:
        path = os.path.join(results_path, "models", f"{last_epoch:06d}",
                            f"model_{variable_name}")
        model = tf.keras.models.load_model(path)
        models.append(model)

    # ------------------------------------------------------------------------

    # Plot the predicted, analytical, and error values for each model as a
    # function of time, at the training points.

    # Plot for each model.
    for iv in range(p.n_var):
        variable_name = p.dependent_variable_names[iv]
        variable_label = p.dependent_variable_labels[iv]
        if verbose:
            print(f"Creating training point PAE plots for {variable_name}.")

        # Create a directory for the PAE plots for this variable.
        pae_path = os.path.join(output_path, f"PAE_{variable_name}")
        print(f"pae_path = {pae_path}")
        os.mkdir(pae_path)

        # Compute the PAE values.
        predicted = model(X_train).numpy().reshape(nt, nx, ny)
        analytical = p.analytical_solutions[iv](
            X_train[:, p.it], X_train[:, p.ix], X_train[:, p.iy]
        ).reshape(nt, nx, ny)
        error = predicted - analytical

        # Plot for each training grid time.
        for it in range(nt):

            # Compute the starting and ending index for this time.
            i0 = it*nx*ny
            i1 = i0 + nx*ny

            # Extract the X and Y values for this time.
            X = X_train[i0:i1, p.ix].reshape(nx, ny).T
            Y = X_train[i0:i1, p.iy].reshape(nx, ny).T

            # To get the proper orientation, reshape, transpose.
            P = predicted[it, :].T
            A = analytical[it, :].T
            E = error[it, :].T

            # Create the plot.
            fig = create_PAE_plot(X, Y, P, A, E)
            fig.suptitle(f"{variable_name} predicted, analytical, and error")

            # Save the plot to a PNG file.
            path = os.path.join(pae_path, f"PAE_{it:04d}_{variable_name}.png")
            fig.savefig(path)
            plt.close(fig)

#     # Load the additional data (boundary and initial conditions).
#     path = os.path.join(results_path, "XY_data.dat")
#     XY_data = np.loadtxt(path)

#     # ------------------------------------------------------------------------

#     # Plot the initial conditions as supplied to the models.

#     # Extract the coordinates of the training points at the initial time.
#     n_start = nx*ny
#     txy0 = tf.Variable(XY_data[:n_start, :p.n_dim])
#     x0 = txy0[:, p.ix].numpy()
#     y0 = txy0[:, p.iy].numpy()

#     # Plot the actual and predicted initial magnetic field vectors.
#     B0x_act = XY_data[:n_start, p.n_dim + p.iBx].reshape(n_start)
#     B0y_act = XY_data[:n_start, p.n_dim + p.iBy].reshape(n_start)
#     if multi:
#         pred = model(txy0).numpy()
#         B0x_pred = pred[:, p.iBx].reshape(n_start)
#         B0y_pred = pred[:, p.iBy].reshape(n_start)
#     else:
#         B0x_pred = models[p.iBx](txy0).numpy().reshape(n_start)
#         B0y_pred = models[p.iBy](txy0).numpy().reshape(n_start)
#     B0x_err = B0x_pred - B0x_act
#     B0y_err = B0y_pred - B0y_act
#     B0x_rms = np.sqrt(np.sum(B0x_err**2)/n_start)
#     B0y_rms = np.sqrt(np.sum(B0y_err**2)/n_start)
#     title = f"Magnetic field at t = 0, (Bx, By)_rms = ({B0x_rms:.2e}, {B0y_rms:.2e})"
#     pinn.standard_plots.plot_actual_predicted_B(
#         x0, y0, B0x_act, B0y_act, B0x_pred, B0y_pred, title=title
#     )
#     path = os.path.join(output_path, "B0_act_pred.png")
#     if verbose:
#         print(f"Saving {path}.")
#     plt.savefig(path)
#     plt.close()

#     # ------------------------------------------------------------------------

#     # # Make a movie for each predicted variable. Include the analytical solution
#     # # and the error.

#     # # Compute the heat map tick locations and labels.
#     # HEATMAP_N_X_TICKS = 5
#     # heatmap_x_tick_pos = np.linspace(0, nx - 1, HEATMAP_N_X_TICKS)
#     # heatmap_x_tick_labels = ["%.1f" % (xmin + x/(nx - 1)*(xmax - xmin)) for x in heatmap_x_tick_pos]
#     # HEATMAP_N_Y_TICKS = 5
#     # heatmap_y_tick_pos = np.linspace(0, ny - 1, HEATMAP_N_Y_TICKS)
#     # heatmap_y_tick_labels = ["%.1f" % (ymin + y/(ny - 1)*(ymax - ymin)) for y in heatmap_y_tick_pos]
#     # heatmap_y_tick_labels = list(reversed(heatmap_y_tick_labels))

#     # # Plot parameters.
#     # plot_min = {
#     #     "Bx": -5e-3,
#     #     "By": -5e-3,
#     # }
#     # plot_max = {
#     #     "Bx": 5e-3,
#     #     "By": 5e-3,
#     # }
#     # plot_err_min = {
#     #     "Bx": -1e-3,
#     #     "By": -1e-3,
#     # }
#     # plot_err_max = {
#     #     "Bx": 1e-3,
#     #     "By": 1e-3,
#     # }

#     # # Create and save each frame.
#     # for (iv, variable_name) in enumerate(p.dependent_variable_names):
#     #     if verbose:
#     #         print(f"Creating movie for {variable_name}.")
#     #     xlabel = p.independent_variable_labels[p.ix]
#     #     ylabel = p.independent_variable_labels[p.iy]
#     #     frame_dir = os.path.join(output_path, f"frames_{variable_name}")
#     #     os.mkdir(frame_dir)
#     #     if multi:
#     #         model = models[0]
#     #         Z_trained = model(X_train).numpy()[:, iv].reshape(nt, nx, ny)
#     #     else:
#     #         model = models[iv]
#     #         Z_trained = model(X_train).numpy().reshape(nt, nx, ny)
#     #     Z_analytical = p.analytical_solutions[iv](
#     #         X_train[:, p.it], X_train[:, p.ix], X_train[:, p.iy]).reshape(nt, nx, ny)
#     #     Z_error = Z_trained - Z_analytical
#     #     frames = []
#     #     for it in range(nt):
#     #         i0 = it*nx*ny
#     #         i1 = i0 + nx*ny
#     #         X = X_train[i0:i1, p.ix]
#     #         Y = X_train[i0:i1, p.iy]
#     #         # To get the proper orientation, reshape, transpose, flip.
#     #         Zt = np.flip(Z_trained[it, :].T, axis=0)
#     #         Za = np.flip(Z_analytical[it, :].T, axis=0)
#     #         Ze = np.flip(Z_error[it, :].T, axis=0)
#     #         pinn.standard_plots.plot_actual_predicted_error(
#     #             X, Y, Za, Zt, Ze,
#     #             title=f"{p.dependent_variable_names[iv]}",
#     #             vmin=plot_min[variable_name], vmax=plot_max[variable_name],
#     #             err_vmin=plot_err_min[variable_name], err_vmax=plot_err_max[variable_name],
#     #             x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     #             y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
#     #         )
#     #         path = os.path.join(frame_dir, f"{variable_name}-{it:06}.png")
#     #         if verbose:
#     #             print(f"Saving {path}.")
#     #         plt.savefig(path)
#     #         frames.append(path)
#     #         plt.close()

#     #     # Assemble the frames into a movie.
#     #     frame_pattern = os.path.join(frame_dir, f"{variable_name}-%06d.png")
#     #     movie_file = os.path.join(output_path, f"{variable_name}.mp4")
#     #     args = [
#     #         "ffmpeg", "-r", "10", "-s", "1920x1080",
#     #         "-i", frame_pattern,
#     #         "-vcodec", "libx264", "-crf", "25", "-pix_fmt", "yuv420p",
#     #         movie_file
#     #     ]
#     #     subprocess.run(args)

#     # # ------------------------------------------------------------------------

#     # # Make a movie of a quiver plot of the trained and analytical solutions.

#     # if verbose:
#     #     print("Creating movie for magnetic field.")
#     # frame_dir = os.path.join(output_path, "frames_BxBy")
#     # os.mkdir(frame_dir)
#     # frames = []
#     # for it in range(nt):
#     #     i0 = it*n_start
#     #     i1 = i0 + n_start
#     #     txy = tf.Variable(X_train[i0:i1, :])
#     #     t = X_train[i0:i1, p.it]
#     #     x = X_train[i0:i1, p.ix]
#     #     y = X_train[i0:i1, p.iy]
#     #     Bx_act = p.Bx_analytical(t, x, y)
#     #     By_act = p.By_analytical(t, x, y)
#     #     Bx_pred = models[p.iBx](txy).numpy().reshape(n_start)
#     #     By_pred = models[p.iBy](txy).numpy().reshape(n_start)
#     #     title = f"Magnetic field at t = {t[0]:.3e}"
#     #     pinn.standard_plots.plot_actual_predicted_B(
#     #         x, y, Bx_act, By_act, Bx_pred, By_pred, title=title
#     #     )
#     #     path = os.path.join(frame_dir, f"BxBy-{it:06}.png")
#     #     if verbose:
#     #         print(f"Saving {path}.")
#     #     plt.savefig(path)
#     #     frames.append(path)
#     #     plt.close()

#     # # Assemble the frames into a movie.
#     # frame_pattern = os.path.join(frame_dir, f"BxBy-%06d.png")
#     # movie_file = os.path.join(output_path, "BxBy.mp4")
#     # args = [
#     #     "ffmpeg", "-r", "10", "-s", "1920x1080",
#     #     "-i", frame_pattern,
#     #     "-vcodec", "libx264", "-crf", "25", "-pix_fmt", "yuv420p",
#     #     movie_file
#     # ]
#     # subprocess.run(args)

#     # # ------------------------------------------------------------------------

#     # # Make a movie of the magnetic field intensity.

#     # # Plot parameters.
#     # plot_min = {
#     #     "B": 0.0,
#     # }
#     # plot_max = {
#     #     "B": 5e-3,
#     # }
#     # plot_err_min = {
#     #     "B": -1e-3,
#     # }
#     # plot_err_max = {
#     #     "B": 1e-3,
#     # }

#     # if verbose:
#     #     print("Creating movie for magnetic field intensity.")
#     # frame_dir = os.path.join(output_path, "frames_B")
#     # os.mkdir(frame_dir)
#     # frames = []
#     # for it in range(nt):
#     #     i0 = it*nx*ny
#     #     i1 = i0 + nx*ny
#     #     txy = tf.Variable(X_train[i0:i1, :])
#     #     t = X_train[i0:i1, p.it]
#     #     x = X_train[i0:i1, p.ix]
#     #     y = X_train[i0:i1, p.iy]
#     #     Bx_act = p.Bx_analytical(t, x, y)
#     #     By_act = p.By_analytical(t, x, y)
#     #     B_act = np.flip(np.sqrt(Bx_act**2 + By_act**2).reshape(nx, ny).T, axis=0)
#     #     Bx_pred = models[p.iBx](txy).numpy()
#     #     By_pred = models[p.iBy](txy).numpy()
#     #     B_pred = np.flip(np.sqrt(Bx_pred**2 + By_pred**2).reshape(nx, ny).T, axis=0)
#     #     B_err = B_pred - B_act
#     #     title = f"Magnetic field intensity at t = {t[0]:.3e}"
#     #     pinn.standard_plots.plot_actual_predicted_error(
#     #         x, y, B_act, B_pred, B_err,
#     #         title=title,
#     #         vmin=plot_min['B'], vmax=plot_max['B'],
#     #         err_vmin=plot_err_min['B'], err_vmax=plot_err_max['B'],
#     #         x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     #         y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
#     #     )
#     #     path = os.path.join(frame_dir, f"B-{it:06}.png")
#     #     if verbose:
#     #         print(f"Saving {path}.")
#     #     plt.savefig(path)
#     #     frames.append(path)
#     #     plt.close()

#     # # Assemble the frames into a movie.
#     # frame_pattern = os.path.join(frame_dir, f"B-%06d.png")
#     # movie_file = os.path.join(output_path, "B.mp4")
#     # args = [
#     #     "ffmpeg", "-r", "10", "-s", "1920x1080",
#     #     "-i", frame_pattern,
#     #     "-vcodec", "libx264", "-crf", "25", "-pix_fmt", "yuv420p",
#     #     movie_file
#     # ]
#     # subprocess.run(args)

#     # # ------------------------------------------------------------------------

#     # # Make a movie of the magnetic field divergence.

#     # # Plot parameters.
#     # plot_min = {
#     #     "divB": -1e-3,
#     # }
#     # plot_max = {
#     #     "divB": 1e-3,
#     # }
#     # plot_err_min = {
#     #     "divB": -1e-3,
#     # }
#     # plot_err_max = {
#     #     "divB": 1e-3,
#     # }

#     # if verbose:
#     #     print("Creating movie for magnetic field divergence.")
#     # frame_dir = os.path.join(output_path, "frames_divB")
#     # os.mkdir(frame_dir)
#     # frames = []
#     # for it in range(nt):
#     #     i0 = it*nx*ny
#     #     i1 = i0 + nx*ny
#     #     txy = tf.Variable(X_train[i0:i1, :])
#     #     t = X_train[i0:i1, p.it]
#     #     x = X_train[i0:i1, p.ix]
#     #     y = X_train[i0:i1, p.iy]
#     #     dBx_dx_act = p.dBx_dx_analytical(t, x, y)
#     #     dBy_dy_act = p.dBy_dy_analytical(t, x, y)
#     #     divB_act = dBx_dx_act + dBy_dy_act
#     #     divB_act = divB_act.reshape(nx, ny)
#     #     divB_act = np.flip(divB_act.T, axis=0)
#     #     with tf.GradientTape(persistent=True) as tape1:
#     #         Bx_pred = models[p.iBx](txy)
#     #         By_pred = models[p.iBy](txy)
#     #     dBx_dx_pred = tape1.gradient(Bx_pred, txy)[:, p.ix].numpy()
#     #     dBy_dy_pred = tape1.gradient(Bx_pred, txy)[:, p.iy].numpy()
#     #     divB_pred = dBx_dx_pred + dBy_dy_pred
#     #     divB_pred = divB_pred.reshape(nx, ny)
#     #     divB_pred = np.flip(divB_pred.T, axis=0)
#     #     divB_err = divB_pred - divB_act
#     #     title = f"Magnetic field divergence at t = {t[0]:.3e}"
#     #     pinn.standard_plots.plot_actual_predicted_error(
#     #         x, y, divB_act, divB_pred, divB_err,
#     #         title=title,
#     #         vmin=plot_min['divB'], vmax=plot_max['divB'],
#     #         err_vmin=plot_err_min['divB'], err_vmax=plot_err_max['divB'],
#     #         x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
#     #         y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
#     #     )
#     #     path = os.path.join(frame_dir, f"divB-{it:06}.png")
#     #     if verbose:
#     #         print(f"Saving {path}.")
#     #     plt.savefig(path)
#     #     frames.append(path)
#     #     plt.close()

#     # # Assemble the frames into a movie.
#     # frame_pattern = os.path.join(frame_dir, f"divB-%06d.png")
#     # movie_file = os.path.join(output_path, "divB.mp4")
#     # args = [
#     #     "ffmpeg", "-r", "10", "-s", "1920x1080",
#     #     "-i", frame_pattern,
#     #     "-vcodec", "libx264", "-crf", "25", "-pix_fmt", "yuv420p",
#     #     movie_file
#     # ]
#     # subprocess.run(args)

    # ------------------------------------------------------------------------

    # Return normally.
    return 0


def main() -> None:
    """Driver for command-line version of code.

    This is the main function for the command-line version of this file. It
    processes the command-line arguments and calls the primary script code.

    Parameters
    ----------
    None

    Returns
    -------
    return_code : int
        Return code from primary script code.

    Raises
    ------
    None
    """
    # Create the command-line parser.
    parser = create_command_line_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}")

    # Convert the arguments from Namespace to dict.
    args = vars(args)

    # Call the main program code.
    return_code = pinn1_plots(**args)
    sys.exit(return_code)


if __name__ == "__main__":
    """Begin main program."""
    main()
