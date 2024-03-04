#!/usr/bin/env python

"""Create plots for pinn0 results for fBz problem.

Create plots for pinn0 results for fBz problem.

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
DESCRIPTION = "Create plots for pinn0 results for fBz problem."

# Name of directory to hold output plots
OUTPUT_DIR = "pinn0_plots"

# Name of problem
PROBLEM_NAME = "fBz"

# Number of points to use in comparison plot.
NUM_POINTS = 101

# Plot limits for dependent variables.
ylim = {}
ylim["L"] = [1e-3, 10]
ylim["fBz"] = [-2.0, 2.0]


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
    if args.debug:
        print(f"args = {args}", flush=True)
    debug = args.debug
    verbose = args.verbose
    results_path = args.results_path

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

    # Plot the loss history.

    # Load the data.
    path = os.path.join(results_path, "L_data.dat")
    L_data = np.loadtxt(path)

    # Create the plot.
    plt.clf()
    plt.semilogy(L_data, label="$L_{data}$")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(ylim["L"])
    plt.legend()
    plt.title(f"Data loss")
    plt.grid()

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "L_data.png")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)
    plt.close()

    # ------------------------------------------------------------------------

    # Load the training points.
    path = os.path.join(results_path, "fBz_011.dat")
    XY_data = np.loadtxt(path)

    # Read the data description from the header.
    with open(path, "r") as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = line[2:]
        fields = line.split(" ")
        Bzmin = float(fields[0])
        Bzmax = float(fields[1])
        nBz = int(fields[2])

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

    # Plot the predicted and analytical solutions, error, and training points.
    if verbose:
            print(f"Creating plot for {variable_name}.")
    xlabel = p.independent_variable_labels[p.iBz]
    ylabel = p.dependent_variable_labels[p.ifBz]
    Bz = XY_data[:, 0]
    model = models[0]
    fBz_model = model(Bz).numpy().reshape(nBz)
    fBz_analytical = p.fBz_analytical(Bz)
    fBz_err = fBz_model - fBz_analytical
    rms_err = np.sqrt(np.sum(fBz_err**2)/nBz)
    plt.plot(Bz, fBz_model, label="trained")
    plt.plot(Bz, fBz_analytical, label="analytical")
    plt.plot(Bz, fBz_err, label="error")
    plt.ylim(ylim["fBz"])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    title = f"{ylabel}, RMS err = {rms_err:.2e}"
    plt.title(title)
    path = os.path.join(output_path, "fBz.png")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    """Begin main program."""
    main()
