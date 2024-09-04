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

# Name of problem
PROBLEM_NAME = "lagaris01"

# Name of directory to hold output plots
OUTPUT_DIR = "pinn1_plots"


def create_command_line_parser():
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


def make_loss_plot(L_res: np.ndarray, L_data: np.ndarray, L: np.ndarray,
                   **kwargs):
    """Make a plot of the aggregate L_res, L_dat, and L.

    Make a plot of the aggregate L_res, L_dat, and L.

    Parameters
    ----------
    L_res : np.ndarray, shape (n_epochs,)
        Values of residual loss for each epoch.
    L_data : np.ndarray, shape (n_epochs,)
        Values of data loss for each epoch.
    L : np.ndarray, shape (n_epochs,)
        Values of weighted loss for each epoch.
    kwargs : dict
        dict of additional keyword arguments

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure for plot.

    Raises
    ------
    None
    """
    # Create the figure and Axes.
    fig, ax = plt.subplots()

    # Plot the data.
    ax.semilogy(L_res, label="$L_{res}$")
    ax.semilogy(L_data, label="$L_{data}$")
    ax.semilogy(L, label="$L$")

    # Decorate the plot.
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.legend()
    title = kwargs.get("title", "")
    ax.set_title(title)

    # Return the figure.
    return fig


def make_predicted_analytical_error_plot(
        Y_predicted: np.ndarray, Y_analytical: np.ndarray,
        Y_error: np.ndarray, X_train: np.ndarray, **kwargs):
    """Make a plot of the predicted and analytical solution, and error.

    Make a plot of the predicted and analytical solution, and error.

    Parameters
    ----------
    Y_predicted : np.ndarray, shape (n_train,)
        Predicted solution at each point.
    Y_analytical : np.ndarray, shape (n_train,)
        Analytical solution at each point.
    Y_error : np.ndarray, shape (n_train,)
        Absolute error at each point.
    X_train : np.ndarray, shape (n_train,)
        Independent variable for each training point.
    kwargs : dict
        dict of additional keyword arguments

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure for plot.

    Raises
    ------
    None
    """
    # Create the figure and Axes.
    fig, ax = plt.subplots()

    # Plot the predicted and analytical solutions on the left y-axis.
    ax.plot(X_train, Y_predicted, label="Predicted")
    ax.plot(X_train, Y_analytical, label="Analytical")

    # Plot the error on the right y-axis.
    ax2 = ax.twinx()
    ax2.plot(X_train, Y_error, label="Error")

    # Combine the axes for the legend.
    lines_left, labels_left = ax.get_legend_handles_labels()
    lines_right, labels_right = ax2.get_legend_handles_labels()
    lines = lines_left + lines_right
    labels = labels_left + labels_right

    # Decorate the plot.
    xlabel = kwargs.get("xlabel", "")
    ax.set_xlabel(xlabel)
    ylabel = kwargs.get("ylabel", "")
    ax.set_ylabel(ylabel)
    ax2.set_ylabel("Error")
    ax.grid()
    ax.legend(lines, labels)
    title = kwargs.get("title", "")
    ax.set_title(title)

    # Return the figure.
    return fig


def pinn1_plots(args: dict):
    """Main program code for pinn1 plots.

    This is the main program code for making pinn1 plots. This function can be
    called from other python code.

    Parameters
    ----------
    args : dict
        Dictionary of command-line options and equivalent options passed from
        the calling function.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Convenience variables.
    if args["debug"]:
        print(f"args = {args}")
    debug = args["debug"]
    verbose = args["verbose"]
    results_path = args["results_path"]

    # ------------------------------------------------------------------------

    # Add the run results directory to the module search path.
    sys.path.append(results_path)

    # Import the problem definition from the run results directory.
    p = import_module(PROBLEM_NAME)

    # Compute the path to the output directory, then create it.
    output_path = OUTPUT_DIR
    os.mkdir(output_path)

    # Create the plots in a memory buffer.
    mpl.use("Agg")

    # ------------------------------------------------------------------------

    # Plot the aggregate residual, data, and overall loss histories.

    # Load the data.
    path = os.path.join(results_path, "L_res.dat")
    L_res = np.loadtxt(path)
    path = os.path.join(results_path, "L_data.dat")
    L_data = np.loadtxt(path)
    path = os.path.join(results_path, "L.dat")
    L = np.loadtxt(path)

    # Create the figure.
    title = "Total residual, data, and weighted loss"
    fig = make_loss_plot(L_res, L_data, L, title=title)

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "L.png")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)

    # Close the figure.
    plt.close(fig)

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
        L_data = np.loadtxt(path)
        path = os.path.join(results_path, f"L_{variable_name}.dat")
        L = np.loadtxt(path)

        # Create the figure.
        title = variable_label + " residual, data, and weighted loss"
        fig = make_loss_plot(L_res, L_data, L, title=title)

        # Save the plot to a PNG file.
        path = os.path.join(output_path, f"L_{variable_name}.png")
        if verbose:
            print(f"Saving {path}.")
        plt.savefig(path)

        # Close the figure.
        plt.close(fig)

    # ------------------------------------------------------------------------

    # Plot the predicted and analytical solutions, and the error, for each
    # model.

    # Load the training points.
    path = os.path.join(results_path, "X_train.dat")
    X_train = np.loadtxt(path)
    n_train = len(X_train)

    # Find the epoch of the last trained model.
    last_epoch = pinn.common.find_last_epoch(results_path)

    # Load the trained model for each variable.
    models = []
    for variable_name in p.dependent_variable_names:
        path = os.path.join(results_path, "models", f"{last_epoch:06d}",
                            f"model_{variable_name}")
        model = tf.keras.models.load_model(path)
        models.append(model)

    # Make a plot for each model.
    for (iv, variable_name) in enumerate(p.dependent_variable_names):
        if verbose:
            print(f"Creating plot for {variable_name}.")

        # Compute the predicted and analytical solution at each training point
        # and the resulting error.
        Y_predicted = models[iv](X_train).numpy().reshape(n_train,)
        Y_analytical = p.Y_analytical[iv](X_train)
        Y_error = Y_predicted - Y_analytical

        # Create the figure.
        title = variable_label + " predicted, analytical and error"
        xlabel = p.independent_variable_labels[0]
        ylabel = p.dependent_variable_labels[iv]
        fig = make_predicted_analytical_error_plot(
            Y_predicted, Y_analytical, Y_error, X_train,
            title=title, xlabel=xlabel, ylabel=ylabel
        )

        # Save the plot to a PNG file.
        path = os.path.join(output_path, f"{variable_name}.png")
        if verbose:
            print(f"Saving {path}.")
        plt.savefig(path)

        # Close the figure.
        plt.close(fig)


def main():
    """Main program code for the command-line version of the script.

    This is the main program code for the command-line version of the script.
    It processes command-line options, then calls the general-purpose entry
    point.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Set up the command-line parser.
    parser = create_command_line_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}")

    # ------------------------------------------------------------------------

    # Call the main program logic. Note that the Namespace object (args)
    # returned from the option parser is converted to a dict using vars().
    pinn1_plots(vars(args))


if __name__ == "__main__":
    """Begin main program."""
    main()
