#!/usr/bin/env python

"""Create plots of ppinn1 results for the alpha_constant problem.

Create plots of ppinn1 results for the alpha_constant problem.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

# Import standard modules.
# import argparse
from importlib import import_module
import os
import shutil
# import subprocess
import sys

# Import supplemental modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import project modules.
from pinn import common
# import pinn.standard_plots


# Program constants

# Program description
DESCRIPTION = "Create plots for pinn1 results for the alpha_constant problem."

# Name of problem
PROBLEM_NAME = "alpha_constant"

# Name of directory to hold output plots
OUTPUT_DIR = "ppinn1_plots"


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
    # Create the standard argument parser for neural network code.
    parser = common.create_minimal_command_line_argument_parser(
        DESCRIPTION
    )

    # Add arguments specific to this script.
    parser.add_argument(
        "--clobber", action="store_true",
        help="Overwrite existing output directory (default: %(default)s)"
    )
    parser.add_argument(
        "results_path",
        help="Path to directory containing results to plot."
    )

    # Return the parser.
    return parser


def make_loss_plot(L: np.ndarray):
    """Make a loss plot.

    Make a standard loss plot.

    Parameters
    ----------
    L: np.ndarray, shape (n_epochs,)
        Array of loss values.

    Returns
    -------
    fig : mpl.figure.Figure
        Figure for plot

    Raises
    ------
    None
    """
    # Create the figure.
    fig, ax = plt.subplots()

    # Create the plot.
    ax.semilogy(L, label="$L$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.legend()
    ax.set_title("Loss")
    ax.grid()

    # Return the figure.
    return fig


def make_t0_Bx_plot(Bx: np.ndarray):
    """Make a contour plot of Bx at (t, z)=(0, 0).

    Make a contour plot of Bx at (t, z)=(0, 0).

    Parameters
    ----------
    Bx: np.ndarray, shape (n_epochs,)
        Array of loss values.

    Returns
    -------
    fig : mpl.figure.Figure
        Figure for plot

    Raises
    ------
    None
    """
    # Create the figure.
    fig, ax = plt.subplots()


def ppinn1_plots(args: dict):
    """Make standard pinn1 plots for alpha_constant.

    Make standard pinn1 plots for alpha_constant.

    Parameters
    ----------
    args: dict
        Dictionary of command-line options.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Local convenience variables
    clobber = args.get("clobber", False)
    debug = args.get("debug", False)
    verbose = args.get("verbose", False)
    results_path = args.get("results_path", None)

    # Basic sanity checks.
    assert results_path is not None

    # ------------------------------------------------------------------------

    # Add the run results directory to the module search path.
    sys.path.append(results_path)

    # Import the problem definition from the run results directory.
    p = import_module(PROBLEM_NAME)

    # Compute the path to the output directory. Then create it if needed.
    output_path = OUTPUT_DIR
    if os.path.isdir(output_path) and clobber:
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    # Create the plots in a memory buffer.
    mpl.use("Agg")

    # ------------------------------------------------------------------------

    # Plot the per-epoch loss history.

    # Load the data.
    path = os.path.join(results_path, "Le.dat")
    L = np.loadtxt(path)

    # Create the plot.
    fig = make_loss_plot(L[:, -1])

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "L.png")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)
    plt.close()

    # ------------------------------------------------------------------------

    # Load the training points.
    path = os.path.join(results_path, "X_train.dat")
    X_train = np.loadtxt(path)
    nx = ny = nz = 50
    nynz = ny*nz
    x = X_train[::nynz, 0]
    y = X_train[:nynz:nz, 1]
    X, Y = np.meshgrid(x, y)

    # Find the epoch of the last trained model.
    last_epoch = common.find_last_epoch(results_path)

    # Load the trained model.
    models = []
    for (iv, variable_name) in enumerate(p.dependent_variable_names):
        path = os.path.join(results_path, "models", f"{last_epoch:06d}",
                            f"model_{variable_name}")
        model = tf.keras.models.load_model(path)
        models.append(model)
    pmodels = []
    for (iv, parameter_name) in enumerate(p.parameter_names):
        path = os.path.join(results_path, "models", f"{last_epoch:06d}",
                            f"parameter_{parameter_name}")
        model = tf.keras.models.load_model(path)
        pmodels.append(model)

    # ------------------------------------------------------------------------

    # Plot the estimated B components at (t, z) = (0, 0).

    # Compute the trained model and parameter values at z=0.
    Yp = [model(X_train) for model in models]
    Pp = [model(X_train) for model in pmodels]

    # Make the Bx plot.
    BX = Yp[0].numpy()[:nx*ny, 0].reshape(ny, nx).T
    cs = plt.contour(X, Y, BX)
    plt.clabel(cs)
    plt.gca().set_aspect(1.0)
    plt.title("$B_x$ at z = 0")
    path = os.path.join(output_path, "Bx.png")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)
    plt.close()

    # Make the By plot.
    BY = Yp[1].numpy()[:nx*ny, 0].reshape(ny, nx).T
    cs = plt.contour(X, Y, BY)
    plt.clabel(cs)
    plt.gca().set_aspect(1.0)
    plt.title("$B_y$ at z = 0")
    path = os.path.join(output_path, "By.png")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)
    plt.close()

    # Make the Bz plot.
    BZ = Yp[2].numpy()[:nx*ny, 0].reshape(ny, nx).T
    cs = plt.contour(X, Y, BZ)
    plt.clabel(cs)
    plt.gca().set_aspect(1.0)
    plt.title("$B_z$ at z = 0")
    path = os.path.join(output_path, "Bz.png")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)
    plt.close()

    # Make the alpha plot.
    ALPHA = Pp[0].numpy()[:nx*ny, 0].reshape(ny, nx).T
    cs = plt.contour(X, Y, ALPHA)
    plt.clabel(cs)
    plt.gca().set_aspect(1.0)
    plt.title(r"$\alpha$ at z = 0")
    path = os.path.join(output_path, "alpha.png")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)
    plt.close()


def main():
    """Driver for command-line version of code."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}")

    # Pass the command-line arguments to the main function as a dict.
    args = vars(args)
    ppinn1_plots(args)


if __name__ == "__main__":
    main()
