#!/usr/bin/env python


"""Create plots for pinn1 results for loop2d_nPuxuyuzBxByBz problem.

Create plots for pinn1 results for loop2d_nPuxuyuzBxByBz problem.

Note on notation: "PAE" -> predicted/analytical/error

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

# Import standard modules.
import copy
from importlib import import_module
import os
import shutil
import subprocess
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
DESCRIPTION = (
    "Create plots for pinn1 results for loop2d_nPuxuyuzBxByBz problem."
)

# Default values for command-line arguments.
DEFAULT_ARGUMENTS = {
    "debug": False,
    "verbose": False,
    "results_path": None,
}

# Name of problem
PROBLEM_NAME = "loop2d_nPuxuyuzBxByBz"

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
    parser = common.create_minimal_command_line_argument_parser(
        DESCRIPTION)
    parser.add_argument(
        "--clobber", action="store_true",
        help="Overwrite existing output directory (default: %(default)s)"
    )
    parser.add_argument(
        "results_path",
        help="Path to directory containing results to plot."
    )
    return parser


def make_loss_plot(L_res: np.ndarray, L_dat: np.ndarray, L: np.ndarray,
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
    title = kwargs.get("title", "")
    figsize = kwargs.get("figsize", None)

    # Create the figure and Axes.
    fig, ax = plt.subplots()

    # Plot the data.
    ax.semilogy(L_res, label="$L_{res}$")
    ax.semilogy(L_dat, label="$L_{dat}$")
    ax.semilogy(L, label="$L$")

    # Decorate the plot.
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.legend()
    ax.set_title(title)

    # Return the figure.
    return fig


def make_PAE_plot(
        Zp: np.ndarray, Za: np.ndarray, Ze: np.ndarray,
        X: np.ndarray, Y: np.ndarray, **kwargs):
    """Make a plot of the predicted and analytical solution, and error.

    Make a plot of the predicted and analytical solution, and error.

    All array arguments must have the same shape (ny, nx).

    Parameters
    ----------
    Zp : np.ndarray, shape (ny, nx)
        Predicted solution at each point.
    Za : np.ndarray, shape (ny, nx)
        Analytical solution at each point.
    Ze : np.ndarray, shape (ny, nx)
        Absolute error (Zp - Za) at each point.
    X : np.ndarray, shape (ny, nx)
        x-values for each training point.
    Y : np.ndarray, shape (ny, nx)
        y-values for each training point.
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
    title = kwargs.get("title", "")
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    figsize = kwargs.get("figsize", None)

    # Create the figure and Axes.
    fig, axs = plt.subplots(1, 3, layout="constrained", figsize=figsize)
    axp, axa, axe = axs

    # Plot the predicted, analytical, and error solutions.
    pcmp = axs[0].pcolormesh(X, Y, Zp)
    axp.set_title("Predicted")
    axp.set_xlabel(xlabel)
    axp.set_ylabel(ylabel)
    fig.colorbar(pcmp, ax=axp, orientation="horizontal")

    pcma = axs[1].pcolormesh(X, Y, Za)
    axa.set_title("Analytical")
    axa.set_xlabel(xlabel)
    axa.set_ylabel(ylabel)
    # Use the same color scale as the predicted solution.
    fig.colorbar(pcmp, ax=axa, orientation="horizontal")

    pcme = axs[2].pcolormesh(X, Y, Ze)
    axe.set_title("Error")
    axe.set_xlabel(xlabel)
    axe.set_ylabel(ylabel)
    fig.colorbar(pcme, ax=axe, orientation="horizontal")

    # Set common plot options.
    for ax in axs:
        ax.grid()
        ax.set_aspect("equal")

    # Decorate the plot.
    fig.suptitle(title)

    # Return the figure.
    return fig


def make_PAE_B_plot(
        Bxp: np.ndarray, Byp: np.ndarray,
        Bxa: np.ndarray, Bya: np.ndarray,
        Bxe: np.ndarray, Bye: np.ndarray,
        X: np.ndarray, Y: np.ndarray, **kwargs):
    """Make a plot of the predicted and analytical B field, and error.

    Make a plot of the predicted and analytical magnetic field, and error.

    All array arguments must have the same shape (ny, nx).

    Parameters
    ----------
    Bxp : np.ndarray, shape (ny, nx)
        Predicted value of Bx.
    Byp : np.ndarray, shape (ny, nx)
        Predicted value of By.
    Bxa : np.ndarray, shape (ny, nx)
        Analytical value of Bx.
    Bya : np.ndarray, shape (ny, nx)
        Analytical value of By.
    Bxe : np.ndarray, shape (ny, nx)
        Absolute error (Bxp - Bxa) at each point.
    Bye : np.ndarray, shape (ny, nx)
        Absolute error (Byp - Bya) at each point.
    X : np.ndarray, shape (ny, nx)
        x-values for each training point.
    Y : np.ndarray, shape (ny, nx)
        y-values for each training point.
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
    title = kwargs.get("title", "")
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    figsize = kwargs.get("figsize", None)

    # Create the figure and Axes.
    fig, axs = plt.subplots(1, 3, layout="constrained", figsize=figsize)
    axp, axa, axe = axs

    # Set common plot options.
    for ax in axs:
        ax.grid()
        ax.set_aspect("equal")

    # Decorate the plot.
    fig.suptitle(title)

    # Return the figure.
    return fig


def make_PAE_divB_plot(
        dBxdxp: np.ndarray, dBydyp: np.ndarray, dBzdzp: np.ndarray,
        dBxdxa: np.ndarray, dBydya: np.ndarray, dBzdza: np.ndarray,
        dBxdxe: np.ndarray, dBydye: np.ndarray, dBzdze: np.ndarray,
        X: np.ndarray, Y: np.ndarray, **kwargs):
    """Make a plot of the predicted and analytical div B, and error.

    Make a plot of the predicted and analytical magnetic field divergence,
    and error.

    All array arguments must have the same shape (ny, nx).

    Parameters
    ----------
    dBxdxp : np.ndarray, shape (ny, nx)
        Predicted value of dBx/dx.
    dBydyp : np.ndarray, shape (ny, nx)
        Predicted value of dBy/dy.
    dBzdzp : np.ndarray, shape (ny, nx)
        Predicted value of dBz/dz.
    dBxdxa : np.ndarray, shape (ny, nx)
        Analytical value of dBx/dx.
    dBydya : np.ndarray, shape (ny, nx)
        Analytical value of dBy/dy.
    dBzdza : np.ndarray, shape (ny, nx)
        Analytical value of dBz/dz.
    dBxdxe : np.ndarray, shape (ny, nx)
        Absolute error (dBxdxp - dBxdxa) at each point.
    dBydye : np.ndarray, shape (ny, nx)
        Absolute error (dBydyp - dBydya) at each point.
    dBzdze : np.ndarray, shape (ny, nx)
        Absolute error (dBzdzp - dBzdza) at each point.
    X : np.ndarray, shape (ny, nx)
        x-values for each training point.
    Y : np.ndarray, shape (ny, nx)
        y-values for each training point.
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
    title = kwargs.get("title", "")
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    figsize = kwargs.get("figsize", None)

    # Create the figure and Axes.
    fig, axs = plt.subplots(1, 3, layout="constrained", figsize=figsize)
    axp, axa, axe = axs

    # Set common plot options.
    for ax in axs:
        ax.grid()
        ax.set_aspect("equal")

    # Decorate the plot.
    fig.suptitle(title)

    # Return the figure.
    return fig


def make_PAE_PB_plot(
        Bxp: np.ndarray, Byp: np.ndarray, Bzp: np.ndarray,
        Bxa: np.ndarray, Bya: np.ndarray, Bza: np.ndarray,
        Bxe: np.ndarray, Bye: np.ndarray, Bze: np.ndarray,
        X: np.ndarray, Y: np.ndarray, **kwargs):
    """Make a plot of the predicted and analytical B pressure, and error.

    Make a plot of the predicted and analytical magnetic pressure, and error.

    All array arguments must have the same shape (ny, nx).

    Parameters
    ----------
    Bxp : np.ndarray, shape (ny, nx)
        Predicted value of Bx.
    Byp : np.ndarray, shape (ny, nx)
        Predicted value of By.
    Bzp : np.ndarray, shape (ny, nx)
        Predicted value of Bz.
    Bxa : np.ndarray, shape (ny, nx)
        Analytical value of Bx.
    Bya : np.ndarray, shape (ny, nx)
        Analytical value of By.
    Bza : np.ndarray, shape (ny, nx)
        Analytical value of Bz.
    Bxe : np.ndarray, shape (ny, nx)
        Absolute error (Bxp - Bxa) at each point.
    Bye : np.ndarray, shape (ny, nx)
        Absolute error (Byp - Bya) at each point.
    Bze : np.ndarray, shape (ny, nx)
        Absolute error (Bzp - Bza) at each point.
    X : np.ndarray, shape (ny, nx)
        x-values for each training point.
    Y : np.ndarray, shape (ny, nx)
        y-values for each training point.
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
    title = kwargs.get("title", "")
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    figsize = kwargs.get("figsize", None)

    # Create the figure and Axes.
    fig, axs = plt.subplots(1, 3, layout="constrained", figsize=figsize)
    axp, axa, axe = axs

    # Set common plot options.
    for ax in axs:
        ax.grid()
        ax.set_aspect("equal")

    # Decorate the plot.
    fig.suptitle(title)

    # Return the figure.
    return fig


def pinn1_plots(args: dict):
    """Primary entry point for pinn1 plots.

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
    # Use defaults for unspecified arguments. Merge in additional settings
    # which are passed in by the caller.
    local_args = copy.deepcopy(DEFAULT_ARGUMENTS)
    if args is not None:
        local_args.update(args)
    args = local_args

    # Local convenience variables
    clobber = args["clobber"]
    debug = args["debug"]
    verbose = args["verbose"]
    results_path = args["results_path"]

    # ------------------------------------------------------------------------

    # Add the run results directory at the head of the module search path.
    sys.path.insert(0, results_path)

    # Import the problem definition from the run results directory.
    p = import_module(PROBLEM_NAME)

    # Compute the path to the output directory, then create it.
    output_path = OUTPUT_DIR
    if os.path.isdir(output_path) and clobber:
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    # ------------------------------------------------------------------------

    # Load the data.

    # Load the loss data.
    if verbose:
        print("Loading loss data.")
    path = os.path.join(results_path, "L_res.dat")
    L_res = np.loadtxt(path)
    path = os.path.join(results_path, "L_dat.dat")
    L_dat = np.loadtxt(path)
    path = os.path.join(results_path, "L.dat")
    L = np.loadtxt(path)

    # Extract the training grid description and data.
    path = os.path.join(results_path, "X_train.dat")
    column_descriptions, X_train = common.read_grid_file(path)

    # NOTE: Column order is: t x y
    # y varies fastest.
    cname = column_descriptions["name"]
    cmin = column_descriptions["min"]
    cmax = column_descriptions["max"]
    cn = column_descriptions["n"]
    n_train = len(X_train)

    # Extract grid counts.
    nt, nx, ny = cn
    nxy = nx*ny  # Number of xy grid points at each time step

    # Extract and reshape the x- and y-grid values from the fitrs time step.
    # It is assumed to be the same for all other time steps.
    # NOTE: Extracted values of data must also use this reshape(ny, nx).T.
    X = X_train[:nxy, 1].reshape(ny, nx).T
    Y = X_train[:nxy, 2].reshape(ny, nx).T

    # ------------------------------------------------------------------------

    # Load the final trained models.

    # Find the epoch of the last trained model.
    last_epoch = common.find_last_epoch(results_path)

    # Load the trained models.
    models = []
    for variable_name in p.dependent_variable_names:
        path = os.path.join(results_path, "models", f"{last_epoch:06d}",
                            f"model_{variable_name}")
        model = tf.keras.models.load_model(path)
        models.append(model)

    # ------------------------------------------------------------------------

    # Create the plots in a memory buffer.
    mpl.use("Agg")

    # ------------------------------------------------------------------------

    # Plot the aggregate losses.

    # Create the figure.
    if verbose:
        print("Creating aggregate loss plot.")
    title = "Total residual, data, and weighted loss"
    fig = make_loss_plot(L_res[:, -1], L_dat[:, -1], L[:, -1], title=title)

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "L.png")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)

    # Close the figure.
    plt.close(fig)

    # ------------------------------------------------------------------------

    # Plot the per-model losses.
    if verbose:
        print("Creating per-model loss plots.")

    # Plot for each model.
    for iv in range(p.n_var):

        # Extract the data.
        variable_name = p.dependent_variable_names[iv]
        variable_label = p.dependent_variable_labels[iv]

        # Create the figure.
        title = f"{variable_label} residual, data, and weighted loss"
        fig = make_loss_plot(L_res[:, iv], L_dat[:, iv], L[:, iv],
                             title=title)

        # Save the plot to a PNG file.
        path = os.path.join(output_path, f"L_{variable_name}.png")
        if verbose:
            print(f"Saving {path}.")
        plt.savefig(path)

        # Close the figure.
        plt.close(fig)

        # End of variable loop.

    # ------------------------------------------------------------------------

    # Make a movie of the predicted and analytical solutions, and the error,
    # for each model.

    # Compute the number of frames in each movie.
    # Assumes same xy points used at each time.
    n_frames = n_train//nxy

    # Constant plot parameters
    figsize = (12, 5)
    xlabel = p.independent_variable_labels[1]
    ylabel = p.independent_variable_labels[2]
    frame_rate = 2  # frames/second

    # Make a plot for each model.
    for iv in range(p.n_var):

        # Extract the variable name and label.
        variable_name = p.dependent_variable_names[iv]
        variable_label = p.dependent_variable_labels[iv]

        if verbose:
            print("Creating predicted/analytical/error movie for "
                  f"{variable_name}.")

        # Compute the predicted and analytical solution at each training point
        # and the resulting error.
        # reshape() needed since TF results are shape (n_train, 1).
        Yp = models[iv](X_train).numpy().reshape(n_train,)
        Ya = p.Y_analytical[iv](X_train[:, 0], X_train[:, 1], X_train[:, 2])
        Ye = Yp - Ya

        # Create the directory for the frames for this variable.
        frame_dir = os.path.join(output_path, f"frames_{variable_name}")
        os.mkdir(frame_dir)

        # Create the frames.
        if verbose:
            print(f"Creating frames for {variable_name}.")
        for i_frame in range(n_frames):

            # Extract the time of the frame.
            t = X_train[i_frame*nxy, 0]

            # Extract the data for the frame.
            i1 = i_frame*nxy
            i2 = (i_frame + 1)*nxy
            # NOTE: Needs same reshape().T used by grid points,.
            Zp = Yp[i1:i2].reshape(ny, nx).T
            Za = Ya[i1:i2].reshape(ny, nx).T
            Ze = Ye[i1:i2].reshape(ny, nx).T

            # Compute the frame title.
            title = f"{variable_label} at t = {t:0.3E}"
            fig = make_PAE_plot(
                Zp, Za, Ze, X, Y,
                title=title, xlabel=xlabel, ylabel=ylabel, figsize=figsize
            )

            # Save the plot to a PNG file.
            path = os.path.join(
                frame_dir, f"{variable_name}-{i_frame:06d}.png")
            if verbose:
                print(f"Saving {path}.")
            plt.savefig(path)

            # Close the figure.
            plt.close(fig)

            # End of frame loop.

        # Assemble the frames into a movie.
        if verbose:
            print(f"Assembling for {variable_name}.")
        frame_pattern = os.path.join(frame_dir, f"{variable_name}-%06d.png")
        movie_file = os.path.join(output_path, f"{variable_name}.mp4")
        cmd = (
            f"ffmpeg -r {frame_rate} -i {frame_pattern} -vcodec libx264 "
            f"-crf 25 -pix_fmt yuv420p {movie_file}"
        )
        subprocess.run(cmd, shell=True, check=True)

        # End of variable loop.

    # ------------------------------------------------------------------------

    # Make a movie of the predicted and analytical magnetic field vector, and
    # the error.
    if verbose:
        print("Creating predicted/analytical/error movie for xy-magnetic "
              "field.")
    title = "Magnetic Field"
    fig = make_PAE_B_plot(
        None, None, None, None, None, None, None, None
    )

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "BxBy.png")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)

    # Close the figure.
    plt.close(fig)

    # ------------------------------------------------------------------------

    # Make a movie of the predicted and analytical magnetic divergence, and
    # the error.
    if verbose:
        print("Creating predicted/analytical/error movie for magnetic "
              "divergence.")
    title = "Magnetic Divergence"
    fig = make_PAE_divB_plot(
        None, None, None, None, None, None, None, None, None, None, None
    )

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "divB.png")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)

    # Close the figure.
    plt.close(fig)

    # ------------------------------------------------------------------------

    # Make a movie of the predicted and analytical magnetic pressure, and
    # the error.
    if verbose:
        print("Creating predicted/analytical/error movie for magnetic "
              "pressure.")
    title = "Magnetic Pressure"
    fig = make_PAE_PB_plot(
        None, None, None, None, None, None, None, None, None, None, None
    )

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "PB.png")
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
