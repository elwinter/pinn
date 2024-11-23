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
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)

    # Plot the data.
    ax.semilogy(L, label="$L$")

    # Decorate the plot.
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.set_title(title)

    # Return the figure.
    return fig


def make_PADE_plot(
        xp: np.ndarray, yp: np.ndarray,
        xa: np.ndarray, ya: np.ndarray,
        xd: np.ndarray, yd: np.ndarray,
        xe: np.ndarray, ye: np.ndarray,
        **kwargs):
    """Make a plot of the predicted and analytical solution, data, and error.

    Make a plot of the predicted and analytical solution, data, and error.

    Parameters
    ----------
    xp : np.ndarray, shape (np,)
        Independent variable for each predicted point.
    yp : np.ndarray, shape (np,)
        Predicted solution at each point.
    xa : np.ndarray, shape (na,)
        Independent variable for each analytical point.
    ya : np.ndarray, shape (na,)
        Analytical solution at each point.
    xd : np.ndarray, shape (nd,)
        Independent variable for each data point.
    yd : np.ndarray, shape (nd,)
        Data at each point.
    xe : np.ndarray, shape (ne,)
        Independent variable for each error point.
    ye : np.ndarray, shape (ne,)
        Error = predicted - analytical at each point.
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
    ax.plot(xp, yp, label="Predicted", color=colors[0])
    ax.plot(xa, ya, label="Analytical", color=colors[1])
    ax.scatter(xd, yd, label="Data", color=colors[2], marker="o")

    # Plot the error on the right y-axis.
    ax2 = ax.twinx()
    ax2.plot(xe, ye, label="Absolute error", color=colors[3])

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

    # Load the training data description and data.
    path = os.path.join(results_path, "XY_data.dat")
    if verbose:
        print(f"Loading training data from {path}.")
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

    # Extract independent and dependent data values.
    Xd = XY_data[:, :p.n_dim]
    Yd = XY_data[:, p.n_dim:]
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
    Yp = np.hstack([model(Xd).numpy() for model in models])
    Ya = np.hstack([f(Xd) for f in p.Y_analytical])
    Ye = Yp - Ya

    # ------------------------------------------------------------------------

    # Create the plots in a memory buffer.
    mpl.use("Agg")

    # ------------------------------------------------------------------------

    # Plot loss histories.

    # Plot the loss history for each model.
    for (iv, variable_name) in enumerate(p.dependent_variable_names):

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

    # Plot the predicted and analytical solutions, data, and error, for each
    # model.

    # Make a plot for each model.
    for (iv, variable_name) in enumerate(p.dependent_variable_names):
        if verbose:
            print("Creating predicted/analytical/data/error plot for "
                  f"{variable_name}.")

        # Create the figure.
        variable_label = p.dependent_variable_labels[iv]
        title = (
            f"pinn0 predicted, analytical, data, and error for {PROBLEM_NAME} "
            f"{variable_label}"
        )
        xlabel = p.independent_variable_labels[p.ix]
        ylabel = variable_label
        fig = make_PADE_plot(
            Xd[:, p.ix], Yp[:, iv], Xd[:, p.ix], Ya[:, iv],
            Xd[:, p.ix], Yd[:, iv], Xd[:, p.ix], Ye[:, iv],
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
