#!/usr/bin/env python

"""Create plots for pinn0 results for lagaris01 problem.

Create plots for pinn0 results for lagaris01 problem.

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
DESCRIPTION = "Create plots for pinn0 results for lagaris01 problem."

# Name of directory to hold output plots
OUTPUT_DIR = "plots_0"

# Name of problem
PROBLEM_NAME = "lagaris01"

# Number of points to use in comparison plot.
NUM_POINTS = 101


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

    # -------------------------------------------------------------------------

    # Plot the data loss history.

    # Read the data.
    path = os.path.join(results_path, "L_data.dat")
    L_dat = np.loadtxt(path)

    # Create the plot in a memory buffer.
    mpl.use("Agg")

    # Create the plot. 
    plt.semilogy(L_dat)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim([1e-6, 1])
    plt.title("Data loss")
    plt.grid()

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "L_dat.png")
    plt.savefig(path)

    # Return to standard plotting backend.
    mpl.use("TkAgg")

    # -------------------------------------------------------------------------

    # Plot the trained and analytical solutions, and the error.

    # Create the plot in a memory buffer.
    mpl.use("Agg")

    # Load the trained model.
    last_epoch = pinn.common.find_last_epoch(results_path)
    path = os.path.join(results_path, "models", f"{last_epoch:06d}",
                        f"model_{p.dependent_variable_names[0]}")
    model = tf.keras.models.load_model(path)

    # Create a set of x points for comparison. Use it to compute the trained
    # and analytical solutions, and error. Also compute RMS error.
    x = np.linspace(p.x0, p.x1, NUM_POINTS)
    Yt = model(x).numpy().reshape(NUM_POINTS)
    Ya = p.Ψ_analytical(x)
    Yerr = Yt - Ya
    rms_err = np.sqrt(np.sum(Yerr**2)/NUM_POINTS)

    # Create the plot.
    x_label = p.independent_variable_labels[0]
    y_name = p.dependent_variable_names[0]
    y_label = p.dependent_variable_labels[0]
    plt.plot(x, Yt, label=f"{y_label} (trained)")
    plt.plot(x, Ya, label=f"{y_label} (analytical)")
    plt.plot(x, Yerr, label=f"{y_label} (error)")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.suptitle("Trained and analytical solution, and error")
    plt.title(f"N = {NUM_POINTS}, RMS error = {rms_err:.2e})")
    plt.grid()

    # Save the plot to a PNG file.
    path = os.path.join(output_path, f"{y_name}.png")
    plt.savefig(path)

    # Return to standard plotting backend.
    mpl.use("TkAgg")


if __name__ == "__main__":
    """Begin main program."""
    main()
