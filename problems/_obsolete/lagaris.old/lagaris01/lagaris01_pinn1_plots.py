#!/usr/bin/env python

"""Create plots for pinn1 results for lagaris01 problem.

Create plots for pinn1 results for lagaris01 problem.

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
DESCRIPTION = "Create plots for pinn1 results for lagaris01 problem."

# Name of directory to hold output plots
OUTPUT_DIR = "pinn1_plots"

# Name of problem
PROBLEM_NAME = "lagaris01"

# Number of points to use in comparison plot.
NUM_POINTS = 101

# Plot limits for dependent variables.
ylim = {}
ylim["L"] = [1e-12, 10]
ylim["Ψ"] = [-0.1, 1.3]


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

    # Plot the total residual, data, and weighted loss histories.

    # Load the data.
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

    # Load the additional data.
    path = os.path.join(results_path, "XY_data.dat")
    XY_data = np.loadtxt(path)

    # Read the data description from the header.
    with open(path, "r") as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = line[2:]
        fields = line.split(" ")
        xmin = float(fields[0])
        xmax = float(fields[1])
        nx = int(fields[2])

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

    # Plot each predicted variable, analytical solution, and error.
    for (iv, variable_name) in enumerate(p.dependent_variable_names):
        if verbose:
            print(f"Creating plot for {variable_name}.")
        xlabel = p.independent_variable_labels[p.ix]
        ylabel = p.dependent_variable_labels[iv]
        X = X_train
        model = models[iv]
        Ym = model(X_train).numpy().reshape(nx)
        Ya = p.Ψ_analytical(X_train)
        Ye = Ym - Ya
        rms_err = np.sqrt(np.sum(Ye**2)/nx)
        plt.plot(X, Ym, label="trained")
        plt.plot(X, Ya, label="analytical")
        plt.plot(X, Ye, label="error")
        plt.ylim(ylim[variable_name])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend()
        title = f"{ylabel}, RMS err = {rms_err:.2e}"
        plt.title(title)
        path = os.path.join(output_path, f"{variable_name}.png")
        if verbose:
            print(f"Saving {path}.")
        plt.savefig(path)
        plt.close()


if __name__ == "__main__":
    """Begin main program."""
    main()
