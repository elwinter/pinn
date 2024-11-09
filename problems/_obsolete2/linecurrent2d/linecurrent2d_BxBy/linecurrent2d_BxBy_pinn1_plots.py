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
import subprocess
import sys

# Import supplemental modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import project modules.
import pinn.common
import pinn.standard_plots


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

    Raises
    ------
    None
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

    # Plot the total residual, data, and weighted loss histories.
    # Also plot the constraint loss, if available.

    # Load the data.
    path = os.path.join(results_path, "L_res.dat")
    L_res = np.loadtxt(path)
    path = os.path.join(results_path, "L_constraint.dat")
    L_constraint = None
    if os.path.exists(path):
        use_constraints = True
        L_constraint = np.loadtxt(path)
    L = np.loadtxt(path)
    path = os.path.join(results_path, "L_data.dat")
    L_dat = np.loadtxt(path)
    path = os.path.join(results_path, "L.dat")
    L = np.loadtxt(path)

    # Create the plot.
    plt.clf()
    plt.semilogy(L_res, label="$L_{res}$")
    if use_constraints:
        plt.semilogy(L_constraint, label="$L_{constraint}$")
    plt.semilogy(L_dat, label="$L_{dat}$")
    plt.semilogy(L, label="$L$")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(ylim["L"])
    plt.legend()
    if use_constraints:
        plt.title(f"Total residual, constraint, data, and weighted loss")
    else:
        plt.title(f"Total residual, data, and weighted loss")
    plt.grid()

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "L.png")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)
    plt.close()

    # ------------------------------------------------------------------------

    # Plot the per-model residual, data, and weighted loss histories.

    # Plot for each model.
    for iv in range(p.n_var):

        # Load the data.
        variable_name = p.dependent_variable_names[iv]
        variable_label = p.dependent_variable_labels[iv]
        path = os.path.join(results_path, f"L_res_{variable_name}.dat")
        L_res = np.loadtxt(path)
        path = os.path.join(results_path, f"L_data_{variable_name}.dat")
        L_dat = np.loadtxt(path)
        path = os.path.join(results_path, f"L_{variable_name}.dat")
        L = np.loadtxt(path)

        # Create the plot.
        plt.semilogy(L_res, label="$L_{res}$")
        plt.semilogy(L_dat, label="$L_{dat}$")
        plt.semilogy(L, label="$L$")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(ylim["L"])
        plt.legend()
        plt.title(f"Residual, data, and weighted loss for {variable_label}")
        plt.grid()

        # Save the plot.
        path = os.path.join(output_path, f"L_{variable_name}.png")
        if verbose:
            print(f"Saving {path}.")
        plt.savefig(path)
        plt.close()

    # ------------------------------------------------------------------------

    # Load the training points.
    path = os.path.join(results_path, "X_train.dat")
    X_train = np.loadtxt(path)

    # Read the data description from the training points header.
    with open(path, "r") as f:
        line = f.readline()  # Skip 1st line
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

    # Load the additional data (boundary and initial conditions).
    path = os.path.join(results_path, "XY_data.dat")
    XY_data = np.loadtxt(path)

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

    # Plot the initial conditions as supplied to the models.

    # Extract the coordinates of the training points at the initial time.
    n_start = nx*ny
    txy0 = tf.Variable(XY_data[:n_start, :p.n_dim])
    x0 = txy0[:, p.ix].numpy()
    y0 = txy0[:, p.iy].numpy()

    # Plot the actual and predicted initial magnetic field vectors.
    B0x_act = XY_data[:n_start, p.n_dim + p.iBx].reshape(n_start)
    B0y_act = XY_data[:n_start, p.n_dim + p.iBy].reshape(n_start)
    B0x_pred = models[p.iBx](txy0).numpy().reshape(n_start)
    B0y_pred = models[p.iBy](txy0).numpy().reshape(n_start)
    title = "Magnetic field at t = 0"
    pinn.standard_plots.plot_actual_predicted_B(
        x0, y0, B0x_act, B0y_act, B0x_pred, B0y_pred, title=title
    )
    path = os.path.join(output_path, "B0_act_pred.png")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)
    plt.close()

    # ------------------------------------------------------------------------

    # Make a movie for each predicted variable. Include the analytical solution
    # and the error.

    # Compute the heat map tick locations and labels.
    HEATMAP_N_X_TICKS = 5
    heatmap_x_tick_pos = np.linspace(0, nx - 1, HEATMAP_N_X_TICKS)
    heatmap_x_tick_labels = ["%.1f" % (xmin + x/(nx - 1)*(xmax - xmin)) for x in heatmap_x_tick_pos]
    HEATMAP_N_Y_TICKS = 5
    heatmap_y_tick_pos = np.linspace(0, ny - 1, HEATMAP_N_Y_TICKS)
    heatmap_y_tick_labels = ["%.1f" % (ymin + y/(ny - 1)*(ymax - ymin)) for y in heatmap_y_tick_pos]
    heatmap_y_tick_labels = list(reversed(heatmap_y_tick_labels))

    # Plot parameters.
    plot_min = {
        "Bx": -5e-3,
        "By": -5e-3,
    }
    plot_max = {
        "Bx": 5e-3,
        "By": 5e-3,
    }
    plot_err_min = {
        "Bx": -1e-3,
        "By": -1e-3,
    }
    plot_err_max = {
        "Bx": 1e-3,
        "By": 1e-3,
    }

    # Create and save each frame.
    for (iv, variable_name) in enumerate(p.dependent_variable_names):
        if verbose:
            print(f"Creating movie for {variable_name}.")
        xlabel = p.independent_variable_labels[p.ix]
        ylabel = p.independent_variable_labels[p.iy]
        frame_dir = os.path.join(output_path, f"frames_{variable_name}")
        os.mkdir(frame_dir)
        model = models[iv]
        Z_trained = model(X_train).numpy().reshape(nt, nx, ny)
        Z_analytical = p.analytical_solutions[iv](
            X_train[:, p.it], X_train[:, p.ix], X_train[:, p.iy]).reshape(nt, nx, ny)
        Z_error = Z_trained - Z_analytical
        frames = []
        for it in range(nt):
            i0 = it*nx*ny
            i1 = i0 + nx*ny
            X = X_train[i0:i1, p.ix]
            Y = X_train[i0:i1, p.iy]
            # To get the proper orientation, reshape, transpose, flip.
            Zt = np.flip(Z_trained[it, :].T, axis=0)
            Za = np.flip(Z_analytical[it, :].T, axis=0)
            Ze = np.flip(Z_error[it, :].T, axis=0)
            pinn.standard_plots.plot_actual_predicted_error(
                X, Y, Za, Zt, Ze,
                title=f"{p.dependent_variable_labels[iv]}",
                vmin=plot_min[variable_name], vmax=plot_max[variable_name],
                err_vmin=plot_err_min[variable_name], err_vmax=plot_err_max[variable_name],
                x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
                y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
            )
            path = os.path.join(frame_dir, f"{variable_name}-{it:06}.png")
            if verbose:
                print(f"Saving {path}.")
            plt.savefig(path)
            frames.append(path)
            plt.close()

        # Assemble the frames into a movie.
        frame_pattern = os.path.join(frame_dir, f"{variable_name}-%06d.png")
        movie_file = os.path.join(output_path, f"{variable_name}.mp4")
        args = [
            "ffmpeg", "-r", "10", "-s", "1920x1080",
            "-i", frame_pattern,
            "-vcodec", "libx264", "-crf", "25", "-pix_fmt", "yuv420p",
            movie_file
        ]
        subprocess.run(args)

    # ------------------------------------------------------------------------

    # Make a movie of a quiver plot of the trained and analytical solutions.

    if verbose:
        print("Creating movie for magnetic field.")
    frame_dir = os.path.join(output_path, "frames_BxBy")
    os.mkdir(frame_dir)
    frames = []
    for it in range(nt):
        i0 = it*n_start
        i1 = i0 + n_start
        txy = tf.Variable(X_train[i0:i1, :])
        t = X_train[i0:i1, p.it]
        x = X_train[i0:i1, p.ix]
        y = X_train[i0:i1, p.iy]
        Bx_act = p.Bx_analytical(t, x, y)
        By_act = p.By_analytical(t, x, y)
        Bx_pred = models[p.iBx](txy).numpy().reshape(n_start)
        By_pred = models[p.iBy](txy).numpy().reshape(n_start)
        title = f"Magnetic field at t = {t[0]:.3e}"
        pinn.standard_plots.plot_actual_predicted_B(
            x, y, Bx_act, By_act, Bx_pred, By_pred, title=title
        )
        path = os.path.join(frame_dir, f"BxBy-{it:06}.png")
        if verbose:
            print(f"Saving {path}.")
        plt.savefig(path)
        frames.append(path)
        plt.close()

    # Assemble the frames into a movie.
    frame_pattern = os.path.join(frame_dir, f"BxBy-%06d.png")
    movie_file = os.path.join(output_path, "BxBy.mp4")
    args = [
        "ffmpeg", "-r", "10", "-s", "1920x1080",
        "-i", frame_pattern,
        "-vcodec", "libx264", "-crf", "25", "-pix_fmt", "yuv420p",
        movie_file
    ]
    subprocess.run(args)

    # ------------------------------------------------------------------------

    # Make a movie of the magnetic field intensity.

    # Plot parameters.
    plot_min = {
        "B": 0.0,
    }
    plot_max = {
        "B": 5e-3,
    }
    plot_err_min = {
        "B": -1e-3,
    }
    plot_err_max = {
        "B": 1e-3,
    }

    if verbose:
        print("Creating movie for magnetic field intensity.")
    frame_dir = os.path.join(output_path, "frames_B")
    os.mkdir(frame_dir)
    frames = []
    for it in range(nt):
        i0 = it*nx*ny
        i1 = i0 + nx*ny
        txy = tf.Variable(X_train[i0:i1, :])
        t = X_train[i0:i1, p.it]
        x = X_train[i0:i1, p.ix]
        y = X_train[i0:i1, p.iy]
        Bx_act = p.Bx_analytical(t, x, y)
        By_act = p.By_analytical(t, x, y)
        B_act = np.flip(np.sqrt(Bx_act**2 + By_act**2).reshape(nx, ny).T, axis=0)
        Bx_pred = models[p.iBx](txy).numpy()
        By_pred = models[p.iBy](txy).numpy()
        B_pred = np.flip(np.sqrt(Bx_pred**2 + By_pred**2).reshape(nx, ny).T, axis=0)
        B_err = B_pred - B_act
        title = f"Magnetic field intensity at t = {t[0]:.3e}"
        pinn.standard_plots.plot_actual_predicted_error(
            x, y, B_act, B_pred, B_err,
            title=title,
            vmin=plot_min['B'], vmax=plot_max['B'],
            err_vmin=plot_err_min['B'], err_vmax=plot_err_max['B'],
            x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
            y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
        )
        path = os.path.join(frame_dir, f"B-{it:06}.png")
        if verbose:
            print(f"Saving {path}.")
        plt.savefig(path)
        frames.append(path)
        plt.close()

    # Assemble the frames into a movie.
    frame_pattern = os.path.join(frame_dir, f"B-%06d.png")
    movie_file = os.path.join(output_path, "B.mp4")
    args = [
        "ffmpeg", "-r", "10", "-s", "1920x1080",
        "-i", frame_pattern,
        "-vcodec", "libx264", "-crf", "25", "-pix_fmt", "yuv420p",
        movie_file
    ]
    subprocess.run(args)

    # ------------------------------------------------------------------------

    # Make a movie of the magnetic field divergence.

    # Plot parameters.
    plot_min = {
        "divB": -1e-3,
    }
    plot_max = {
        "divB": 1e-3,
    }
    plot_err_min = {
        "divB": -1e-3,
    }
    plot_err_max = {
        "divB": 1e-3,
    }

    if verbose:
        print("Creating movie for magnetic field divergence.")
    frame_dir = os.path.join(output_path, "frames_divB")
    os.mkdir(frame_dir)
    frames = []
    for it in range(nt):
        i0 = it*nx*ny
        i1 = i0 + nx*ny
        txy = tf.Variable(X_train[i0:i1, :])
        t = X_train[i0:i1, p.it]
        x = X_train[i0:i1, p.ix]
        y = X_train[i0:i1, p.iy]
        dBx_dx_act = p.dBx_dx_analytical(t, x, y)
        dBy_dy_act = p.dBy_dy_analytical(t, x, y)
        divB_act = dBx_dx_act + dBy_dy_act
        divB_act = divB_act.reshape(nx, ny)
        divB_act = np.flip(divB_act.T, axis=0)
        with tf.GradientTape(persistent=True) as tape1:
            Bx_pred = models[p.iBx](txy)
            By_pred = models[p.iBy](txy)
        dBx_dx_pred = tape1.gradient(Bx_pred, txy)[:, p.ix].numpy()
        dBy_dy_pred = tape1.gradient(Bx_pred, txy)[:, p.iy].numpy()
        divB_pred = dBx_dx_pred + dBy_dy_pred
        divB_pred = divB_pred.reshape(nx, ny)
        divB_pred = np.flip(divB_pred.T, axis=0)
        divB_err = divB_pred - divB_act
        title = f"Magnetic field divergence at t = {t[0]:.3e}"
        pinn.standard_plots.plot_actual_predicted_error(
            x, y, divB_act, divB_pred, divB_err,
            title=title,
            vmin=plot_min['divB'], vmax=plot_max['divB'],
            err_vmin=plot_err_min['divB'], err_vmax=plot_err_max['divB'],
            x_tick_pos=heatmap_x_tick_pos, x_tick_labels=heatmap_x_tick_labels,
            y_tick_pos=heatmap_y_tick_pos, y_tick_labels=heatmap_y_tick_labels,
        )
        path = os.path.join(frame_dir, f"divB-{it:06}.png")
        if verbose:
            print(f"Saving {path}.")
        plt.savefig(path)
        frames.append(path)
        plt.close()

    # Assemble the frames into a movie.
    frame_pattern = os.path.join(frame_dir, f"divB-%06d.png")
    movie_file = os.path.join(output_path, "divB.mp4")
    args = [
        "ffmpeg", "-r", "10", "-s", "1920x1080",
        "-i", frame_pattern,
        "-vcodec", "libx264", "-crf", "25", "-pix_fmt", "yuv420p",
        movie_file
    ]
    subprocess.run(args)


if __name__ == "__main__":
    """Begin main program."""
    main()
