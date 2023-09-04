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
    yt = model(x).numpy().reshape(NUM_POINTS)
    ya = p.Ψ_analytical(x)
    yerr = yt - ya
    rms_err = np.sqrt(np.sum(yerr**2)/NUM_POINTS)

    # Create the plot.
    x_label = p.independent_variable_labels[0]
    y_name = p.dependent_variable_names[0]
    y_label = p.dependent_variable_labels[0]
    plt.plot(x, yt, label=f"{y_label} (trained)")
    plt.plot(x, ya, label=f"{y_label} (analytical)")
    plt.plot(x, yerr, label=f"{y_label} (error)")
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

    # -------------------------------------------------------------------------

    # Plot the trained and analytical derivatives, and the error.

    # Create the plot in a memory buffer.
    mpl.use("Agg")

    # Compute the trained and analytical derivatives, and error.
    # Also compute RMS error.
    xv = tf.Variable(x.reshape(NUM_POINTS, 1))
    with tf.GradientTape(persistent=True) as tape1:
        yt = model(xv)
    dyt_dx = tape1.gradient(yt, xv).numpy().reshape(NUM_POINTS)
    dya_dx = p.dΨ_dx_analytical(x)
    dy_dx_err = dyt_dx - dya_dx
    rms_err = np.sqrt(np.sum(dy_dx_err**2)/NUM_POINTS)

    # Create the plot.
    x_label = p.independent_variable_labels[0]
    y_name = f"d{p.dependent_variable_names[0]}_d{p.independent_variable_names[0]}"
    y_label = f"d{p.dependent_variable_labels[0]}/d{p.independent_variable_labels[0]}"
    plt.plot(x, dyt_dx, label=f"{y_label} (trained)")
    plt.plot(x, dya_dx, label=f"{y_label} (analytical)")
    plt.plot(x, dy_dx_err, label=f"{y_label} (error)")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.suptitle("Trained and analytical derivative, and error")
    plt.title(f"N = {NUM_POINTS}, RMS error = {rms_err:.2e})")
    plt.grid()

    # Save the plot to a PNG file.
    path = os.path.join(output_path, f"{y_name}.png")
    plt.savefig(path)

    # Return to standard plotting backend.
    mpl.use("TkAgg")

    # -------------------------------------------------------------------------

    # Plot the weights and biases for each layer.

    # Create the plot in a memory buffer.
    mpl.use("Agg")

    # Input/first hidden layer weights
    i = 0
    layer = model.layers[i]
    w = layer.variables[0].numpy()
    n_inputs = w.shape[0]
    n_nodes = w.shape[1]
    w.shape = (n_nodes,)
    x = np.arange(n_nodes)
    plt.bar(x, w)
    plt.xlabel("Node")
    plt.ylabel("$w$")
    plt.title(f"Layer {i} weights")
    path = os.path.join(output_path, f"w{i:02}.png")
    plt.savefig(path)
    plt.clf()

    # Input layer biases
    i = 0
    layer = model.layers[i]
    b = layer.variables[1].numpy()
    w.shape = (n_nodes,)
    x = np.arange(n_nodes)
    plt.bar(x, b)
    plt.xlabel("Node")
    plt.ylabel("$b$")
    plt.title(f"Layer {i} biases")
    path = os.path.join(output_path, f"b{i:02}.png")
    plt.savefig(path)
    plt.clf()

    # Output layer weights
    i = 1
    layer = model.layers[i]
    w = layer.variables[0].numpy()
    n_inputs = w.shape[0]
    n_nodes = w.shape[1]
    w.shape = (n_inputs,)
    x = np.arange(n_inputs)
    plt.bar(x, w)
    plt.xlabel("Node")
    plt.ylabel("$w$")
    plt.title(f"Layer {i} weights")
    path = os.path.join(output_path, f"w{i:02}.png")
    plt.savefig(path)
    plt.clf()

    # Return to standard plotting backend.
    mpl.use("TkAgg")


if __name__ == "__main__":
    """Begin main program."""
    main()
