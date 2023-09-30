#!/usr/bin/env python

"""Create plots for pinn1 results for bw1d_nPux problem.

Create plots for pinn1 results for bw1d_nPux problem.

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


# Program constants

# Program description
DESCRIPTION = "Create plots for pinn1 results for bw1d_nPux problem."

# Name of directory to hold output plots
OUTPUT_DIR = "pinn1_plots"

# Name of problem
PROBLEM_NAME = "bw1d_nPux"

# Plot limits for dependent variables.
ylim = {}
ylim["L"] = [1e-6, 1]
ylim["n"] = [0, 1.1]
ylim["P"] = [0, 1.1]
ylim["ux"] = [0, 1.1]


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
        "--clobber", action="store_true",
        help="Clobber existing output directory (default: %(default)s)."
    )
    parser.add_argument(
        "-d", "--debug", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "results_path",
        help="Path to directory containing results to plot."
    )
    parser.add_argument(
        "nt", type=int, help="Number of points in t-dimension"
    )
    parser.add_argument(
        "nx", type=int, help="Number of points in x-dimension"
    )
    return parser


def main():
    """Main program."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    clobber = args.clobber
    debug = args.debug
    verbose = args.verbose
    results_path = args.results_path
    nt = args.nt
    nx = args.nx
    if debug:
        print(f"args = {args}", flush=True)

    # Add the run results directory to the module search path.
    sys.path.append(results_path)

    # Import the problem definition from the run results directory.
    p = import_module(PROBLEM_NAME)

    # Compute the path to the output directory. Then create it if needed.
    output_path = OUTPUT_DIR
    if os.path.exists(output_path):
        if not clobber:
            raise FileExistsError(f"Output directory {output_path} exists!")
    else:
        os.mkdir(output_path)

    # Create the plots in a memory buffer.
    mpl.use("Agg")

    # -------------------------------------------------------------------------

    # Plot the aggregate residual, data, and weighted loss histories.

    # Read the data.
    path = os.path.join(results_path, "L_res.dat")
    L_res = np.loadtxt(path)
    path = os.path.join(results_path, "L_data.dat")
    L_dat = np.loadtxt(path)
    path = os.path.join(results_path, "L.dat")
    L = np.loadtxt(path)

    # Create the plot.
    plt.clf()
    plt.semilogy(L_res, label="$L_{res}$")
    plt.semilogy(L_dat, label="$L_{dat}$")
    plt.semilogy(L, label="$L$")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(ylim["L"])
    plt.legend()
    plt.title(f"Aggregate residual, data, and weighted loss")
    plt.grid()

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "L.png")
    plt.savefig(path)
    plt.close()

    # ------------------------------------------------------------------------

    # Plot the per-model residual, data, and weighted loss histories.

    # Plot for each model.
    for v in p.dependent_variable_names:

        # Read the loss histories for this model.
        path = os.path.join(results_path, f"L_res_{v}.dat")
        L_res = np.loadtxt(path)
        path = os.path.join(results_path, f"L_data_{v}.dat")
        L_dat = np.loadtxt(path)
        path = os.path.join(results_path, f"L_{v}.dat")
        L = np.loadtxt(path)

        # Create the plot.
        plt.clf()
        plt.semilogy(L_res, label="$L_{res}$")
        plt.semilogy(L_dat, label="$L_{dat}$")
        plt.semilogy(L, label="$L$")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(ylim["L"])
        plt.legend()
        plt.title(f"Residual, data, and aggregate loss for {v}")
        plt.grid()

        # Save the plot to a PNG file.
        path = os.path.join(output_path, f"L_{v}.png")
        plt.savefig(path)
        plt.close()

    # ------------------------------------------------------------------------

    # Load the training points.
    path = os.path.join(results_path, "X_train.dat")
    X_train = np.loadtxt(path)

    # Load the additional data.
    path = os.path.join(results_path, "XY_data.dat")
    XY_data = np.loadtxt(path)

    # Find the last trained model.
    last_epoch = pinn.common.find_last_epoch(results_path)

    # ------------------------------------------------------------------------

    # Plot the initial pressure provided by data.
    v = p.dependent_variable_names[p.iP]
    xlabel = p.independent_variable_names[p.ix]
    ylabel = p.dependent_variable_labels[p.iP]

    # Load the data for P(0).
    x = XY_data[:, p.ix]
    P = XY_data[:, p.n_dim + p.iP]
    print(P.min(), P.max(), P)
    plt.plot(x, P)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Initial pressure data")
    path = os.path.join("P0.png")
    plt.savefig(path)
    plt.close()

    # ------------------------------------------------------------------------

    # Make a movie of the density evolution.
    v = p.dependent_variable_names[p.i_n]
    xlabel = p.independent_variable_names[p.ix]
    ylabel = p.dependent_variable_labels[p.i_n]

    # Load the model for this variable.
    path = os.path.join(results_path, "models", f"{last_epoch:06d}",
                        f"model_{v}")
    model = tf.keras.models.load_model(path)

    # Evaluate the model at the training points.
    Y_train = model(X_train).numpy().reshape(nt, nx)

    # Create a frame directory for this variable.
    frame_dir = os.path.join(output_path, f"frames_{v}")
    os.mkdir(frame_dir)

    # Create and save each frame.
    frames = []
    for i in range(nt):
        plt.clf()
        i0 = i*nx
        i1 = i0 + nx
        x = X_train[i0:i1, 1]
        y = Y_train[i, :]
        plt.plot(x, y)
        plt.ylim([0, 1.1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        t_frame = X_train[i0, 0]
        t_label = f"t = {t_frame:.2e}"
        plt.text(-1.0, 1.03, t_label)
        title = ylabel
        plt.title(title)
        path = os.path.join(frame_dir, f"{v}-{i:06}.png")
        plt.savefig(path)
        frames.append(path)

    # Assemble the frames into a movie.
    args = ["convert", "-delay", "2", "-loop", "0"]
    args.extend(frames)
    path = os.path.join(output_path, f"{v}.gif")
    args.append(path)
    subprocess.run(args)

    # ------------------------------------------------------------------------

    # Make a movie of the pressure evolution.
    v = p.dependent_variable_names[p.iP]
    xlabel = p.independent_variable_names[p.ix]
    ylabel = p.dependent_variable_labels[p.iP]

    # Load the model for this variable.
    path = os.path.join(results_path, "models", f"{last_epoch:06d}",
                        f"model_{v}")
    model = tf.keras.models.load_model(path)

    # Evaluate the model at the training points.
    Y_train = model(X_train).numpy().reshape(nt, nx)

    # Create a frame directory for this variable.
    frame_dir = os.path.join(output_path, f"frames_{v}")
    os.mkdir(frame_dir)

    # Create and save each frame.
    frames = []
    for i in range(nt):
        plt.clf()
        i0 = i*nx
        i1 = i0 + nx
        x = X_train[i0:i1, 1]
        y = Y_train[i, :]
        plt.plot(x, y)
        plt.ylim([0, 1.1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        t_frame = X_train[i0, 0]
        t_label = f"t = {t_frame:.2e}"
        plt.text(-1.0, 1.03, t_label)
        plt.title(ylabel)
        path = os.path.join(frame_dir, f"{v}-{i:06}.png")
        plt.savefig(path)
        frames.append(path)

    # Assemble the frames into a movie.
    args = ["convert", "-delay", "2", "-loop", "0"]
    args.extend(frames)
    path = os.path.join(output_path, f"{v}.gif")
    args.append(path)
    subprocess.run(args)

    # ------------------------------------------------------------------------

    # Make a movie of the x-velocity evolution.
    v = p.dependent_variable_names[p.iux]
    xlabel = p.independent_variable_names[p.ix]
    ylabel = p.dependent_variable_labels[p.iux]

    # Load the model for this variable.
    path = os.path.join(results_path, "models", f"{last_epoch:06d}",
                        f"model_{v}")
    model = tf.keras.models.load_model(path)

    # Evaluate the model at the training points.
    Y_train = model(X_train).numpy().reshape(nt, nx)

    # Create a frame directory for this variable.
    frame_dir = os.path.join(output_path, f"frames_{v}")
    os.mkdir(frame_dir)

    # Create and save each frame.
    frames = []
    for i in range(nt):
        plt.clf()
        i0 = i*nx
        i1 = i0 + nx
        x = X_train[i0:i1, 1]
        y = Y_train[i, :]
        plt.plot(x, y)
        plt.ylim([0, 1.1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        t_frame = X_train[i0, 0]
        t_label = f"t = {t_frame:.2e}"
        plt.text(-1.0, 1.03, t_label)
        plt.title(ylabel)
        path = os.path.join(frame_dir, f"{v}-{i:06}.png")
        plt.savefig(path)
        frames.append(path)

    # Assemble the frames into a movie.
    args = ["convert", "-delay", "2", "-loop", "0"]
    args.extend(frames)
    path = os.path.join(output_path, f"{v}.gif")
    args.append(path)
    subprocess.run(args)


if __name__ == "__main__":
    """Begin main program."""
    main()
