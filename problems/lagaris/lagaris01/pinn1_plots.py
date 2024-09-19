#!/usr/bin/env python

"""Create plots for pinn1 results for lagaris01 problem.

Create plots for pinn1 results for lagaris01 problem.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

# Import standard modules.
from importlib import import_module
import os
import shutil
import sys

# Import supplemental modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import project modules.
from pinn import common


# Program constants

# Program description
DESCRIPTION = "Create plots for pinn1 results for lagaris01 problem."

# Name of problem
PROBLEM_NAME = "lagaris01"

# Name of directory to hold output plots
OUTPUT_DIR = "pinn1_plots"


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


def make_PAE_plot(X: np.ndarray, Yp: np.ndarray, Ya: np.ndarray,
                  Ye: np.ndarray, **args):
    """Make a predicted-analytical-error plot.

    Make a predicted-analytical-error plot.

    Parameters
    ----------
    X: np.ndarray, shape (n,)
        Array of X values.
    Yp: np.ndarray, shape (n,)
        Array of predicted Y values.
    Ya: np.ndarray, shape (n,)
        Array of analytical Y values.
    Ye: np.ndarray, shape (n,)
        Array of Yp - Ya values.
    args: dict
        Dictionary of additional plot options.

    Returns
    -------
    fig : mpl.figure.Figure
        Figure for plot

    Raises
    ------
    None
    """
    # Create the figure, primary, and secondary axes.
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    # Create the plot.
    xlabel = args.get("xlabel", "x")
    ylabel = args.get("ylabel", "y")
    rms_err = np.sqrt(np.sum(Ye**2)/Ye.shape[0])

    # Plot predicted and analytical values on primary axis.
    line_p = ax.plot(X, Yp, label="predicted")
    line_a = ax.plot(X, Ya, label="analytical")

    # Plot error values on secondary axis.
    line_e = ax2.plot(X, Ye, label="error")

    # Create the legend.
    lines = line_p + line_a + line_e
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)

    # Decorate the plot.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    title = f"{ylabel}, RMS err = {rms_err:.2e}"
    ax.set_title(title)

    # Return the figure.
    return fig


def pinn1_plots(args: dict):
    """Make standard pinn1 plots for lagaris01.

    Make standard pinn1 plots for lagaris01.

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

    # Plot the predicted and analytical solutions, and error.

    # Load the training points.
    path = os.path.join(results_path, "X_train.dat.gz")
    if not os.path.isfile(path):
        path = os.path.join(results_path, "X_train.dat")
    X_train = np.loadtxt(path)

    # Find the epoch of the last trained model.
    last_epoch = common.find_last_epoch(results_path)

    # Load the trained model.
    variable_name = p.dependent_variable_names[p.iΨ]
    path = os.path.join(results_path, "models", f"{last_epoch:06d}",
                        f"model_{variable_name}")
    model = tf.keras.models.load_model(path)

    # Compute predicted, analytical, and error values.
    X = X_train
    Yp = model(X).numpy().reshape(X.shape[0])
    Ya = p.Ψ_analytical(X)
    Ye = Yp - Ya

    # Create the plot.
    fig = make_PAE_plot(
        X, Yp, Ya, Ye,
        xlabel=p.independent_variable_labels[p.ix],
        ylabel=p.dependent_variable_labels[p.iΨ],
    )

    # Save the plot to a PNG file.
    path = os.path.join(output_path, f"{variable_name}.png")
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
    pinn1_plots(args)


if __name__ == "__main__":
    main()
