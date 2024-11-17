#!/usr/bin/env python

"""Create plots for pinn0 results for the lagaris01 problem.

Create plots for pinn0 results for the lagaris01 problem.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

# Import standard modules.
import copy
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
DESCRIPTION = "Create plots for pinn0 results for the lagaris01 problem."

# Default values for command-line arguments.
DEFAULT_ARGUMENTS = {
    "clobber": common.DEFAULT_ARGUMENTS["clobber"],
    "debug": common.DEFAULT_ARGUMENTS["debug"],
    "image_format": "png",
    "verbose": common.DEFAULT_ARGUMENTS["verbose"],
    "results_path": None,
}

# Name of problem
PROBLEM_NAME = "lagaris01"

# Name of directory to hold output plots
OUTPUT_DIR = "pinn0_plots"


def create_command_line_parser():
    """Create the command-line parser.

    Create the command-line parser.

    Parameters
    ----------
    description : str, default DESCRIPTION
        Parser for the command-line.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser for command-line arguments.

    Raises
    ------
    None
    """
    parser = common.create_minimal_command_line_parser(DESCRIPTION)
    parser.add_argument(
        "--clobber", action="store_true",
        help="Overwrite existing output directory (default: %(default)s)"
    )
    parser.add_argument(
        "--image_format", default=DEFAULT_ARGUMENTS["image_format"],
        help="Plot image type extension (png|pdf) (default: %(default)s)"
    )
    parser.add_argument(
        "results_path",
        help="Path to directory containing results to plot."
    )
    return parser


def create_output_directory(clobber: bool = False):
    """Create the output directory for the plots.

    Create the output directory for the plots. The name of the output
    directory is "pinn0_plots".

    Parameters
    ----------
    clobber : bool, default False
        True to delete existing directory of same name.

    Returns
    -------
    output_dir : str
        Path to output directory.

    Raises
    ------
    None
    """
    output_dir = "pinn0_plots"
    if os.path.isdir(output_dir) and clobber:
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    return output_dir


def make_pinn0_loss_plot(L: np.ndarray,  **kwargs):
    """Make a plot of the pinn0 model loss history.

    Make a plot of the pinn0 model loss history.

    Parameters
    ----------
    L : np.ndarray, shape (n_epochs,)
        Values of loss for each epoch.
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
    # Extract optional keywords.
    title = kwargs.get("title", "")
    figsize = kwargs.get("figsize", None)

    # Create the figure and Axes.
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the data.
    ax.semilogy(L, label="$L$")

    # Decorate the plot.
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.set_title(title)

    # Return the figure.
    return fig


def make_PAE_plot(
        Yp: np.ndarray, Ya: np.ndarray, Ye: np.ndarray,
        X: np.ndarray, **kwargs):
    """Make a plot of the predicted and analytical solution, and error.

    Make a plot of the predicted and analytical solution, and error.

    Parameters
    ----------
    Yp : np.ndarray, shape (n,)
        Predicted solution at each point.
    Ya : np.ndarray, shape (n,)
        Analytical solution at each point.
    Ye : np.ndarray, shape (n,)
        Absolute error at each point.
    X : np.ndarray, shape (n,)
        Independent variable for each point.
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
    # Extract optional keywords.
    figsize = kwargs.get("figsize", None)
    title = kwargs.get("title", "")
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")

    # Get the default color cycle.
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    # Create the figure and Axes.
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)

    # Plot the predicted and analytical solutions on the left y-axis.
    ax.plot(X, Yp, label="Predicted", color=colors[0])
    ax.plot(X, Ya, label="Analytical", color=colors[1])

    # Plot the error on the right y-axis.
    ax2 = ax.twinx()
    ax2.plot(X, Ye, label="Absolute error", color=colors[2])

    # Combine the axes for the legend.
    lines_left, labels_left = ax.get_legend_handles_labels()
    lines_right, labels_right = ax2.get_legend_handles_labels()
    lines = lines_left + lines_right
    labels = labels_left + labels_right

    # Decorate the plot.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax2.set_ylabel("Absolute Error")
    ax.grid()
    ax.legend(lines, labels)
    ax.set_title(title)

    # Return the figure.
    return fig


def pinn0_plots(args: dict):
    """Main program code for pinn0 plots.

    This is the main program code for making pinn0 plots. This function can be
    called from other python code.

    Parameters
    ----------
    args : dict
        Dictionary of command-line and other options.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Use defaults for unspecified arguments. Merge in additional settings
    # which are passed in by the caller.
    local_args = copy.deepcopy(DEFAULT_ARGUMENTS)
    if args is not None:
        local_args.update(args)
    args = local_args
    if args["debug"]:
        print(f"args = {args}")

    # Local convenience variables
    clobber = args["clobber"]
    debug = args["debug"]
    image_format = args["image_format"]
    verbose = args["verbose"]
    results_path = args["results_path"]

    # ------------------------------------------------------------------------

    # Add the run results directory to the module search path.
    sys.path.append(results_path)

    # Import the problem definition from the run results directory.
    p = import_module(PROBLEM_NAME)
    if debug:
        print(f"p = {p}")

    # Compute the path to the output directory, then create it.
    if verbose:
        print("Creating output directory.")
    output_dir = create_output_directory(clobber)
    if debug:
        print(f"output_dir = {output_dir}")

    # ------------------------------------------------------------------------

    # Load the data.

    # Loss
    if verbose:
        print("Loading loss history.")
    path = os.path.join(results_path, "L.dat")
    # L is np.ndarray shape (n_epochs,).
    L = np.loadtxt(path)
    if debug:
        print(f"L = {L}")

    # Extract the training data description and data.
    path = os.path.join(results_path, "XY_data.dat")
    # column_names is a list of str, length p.n_var.
    # column_descriptions is a dict of dicts, length p.n_var.
    # XY_data.dat is a np.ndarray of float, shape (n_data, p.n_dim + p.n_var).
    column_names, column_descriptions, XY_data = common.read_grid_file(path)
    if debug:
        print(f"column_names = {column_names}")
        print(f"column_descriptions = {column_descriptions}")
        print(f"XY_data = {XY_data}")

    # Count the data points.
    n_data = XY_data.shape[0]
    if debug:
        print(f"n_data = {n_data}")

    # Extract individual columns.
    Xd = XY_data[:, 0]
    Yd = XY_data[:, 1]
    if debug:
        print(f"Xd = {Xd}")
        print(f"Yd = {Yd}")

    # ------------------------------------------------------------------------

    # Load the final trained models.

    # Find the epoch of the last trained model.
    last_epoch = common.find_last_epoch(results_path)
    if debug:
        print(f"last_epoch = {last_epoch}")

    # Load the trained models.
    models = []
    if verbose:
        print("Loading trained models.")
    for variable_name in p.dependent_variable_names:
        path = os.path.join(results_path, "models", f"{last_epoch:06d}",
                            f"model_{variable_name}")
        model = tf.keras.models.load_model(path)
        models.append(model)
    if debug:
        print(f"models = {models}")

    # ------------------------------------------------------------------------

    # Compute the predicted and analytical solutions, and error.
    if verbose:
        print("Computing predicted and analytical values, and error.")
    Yp = [model(Xd).numpy().reshape(n_data,) for model in models]
    Ya = [f(Xd) for f in p.Y_analytical]
    Ye = [pr - an for (pr, an) in zip(Yp, Ya)]

    # ------------------------------------------------------------------------

    # Create the plots in a memory buffer.
    mpl.use("Agg")

    # ------------------------------------------------------------------------

    # Plot loss histories.

    # Plot the loss history for each model.
    for (iv, variable_name) in enumerate(p.dependent_variable_names):
        if verbose:
            print(f"Creating loss plot for {variable_name}.")

        # Create the figure.
        if verbose:
            print("Creating pinn0 model loss plot for variable "
                  f"{variable_name}.")
        title = (
            f"pinn0 model loss for {PROBLEM_NAME} "
            f"{p.dependent_variable_labels[iv]}"
        )
        fig = make_pinn0_loss_plot(L[:, iv], title=title)

        # Save the plot to a file.
        path = os.path.join(output_dir, f"L_{variable_name}.{image_format}")
        if verbose:
            print(f"Saving {path}.")
        plt.savefig(path)

        # Close the figure.
        plt.close(fig)

    # Plot the aggregate model loss history.
    if verbose:
        print("Creating aggregate loss plot.")
    title = f"pinn0 model loss for {PROBLEM_NAME}"
    fig = make_pinn0_loss_plot(L[:, -1], title=title)

    # Save the plot to a file.
    path = os.path.join(output_dir, f"L.{image_format}")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)

    # Close the figure.
    plt.close(fig)

    # ------------------------------------------------------------------------

    # Plot the predicted and analytical solutions, and the error, for each
    # model.

    # Make a plot for each model.
    for (iv, variable_name) in enumerate(p.dependent_variable_names):
        if verbose:
            print("Creating predicted/analytical/error plot for "
                  f"{variable_name}.")

        # Create the figure.
        variable_label = p.dependent_variable_labels[iv]
        title = (
            f"pinn0 predicted, analytical, error for {PROBLEM_NAME} "
            f"{variable_label}"
        )
        xlabel = p.independent_variable_labels[p.ix]
        ylabel = variable_label
        fig = make_PAE_plot(
            Yp[iv], Ya[iv], Ye[iv], Xd,
            title=title, xlabel=xlabel, ylabel=ylabel
        )

        # Save the plot to a file.
        path = os.path.join(output_dir, f"{variable_name}.{image_format}")
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

    # Convert the arguments from Namespace to dict.
    args = vars(args)

    # Call the main program logic.
    return_code = pinn0_plots(args)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
