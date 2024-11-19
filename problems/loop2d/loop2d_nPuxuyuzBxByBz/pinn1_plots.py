#!/usr/bin/env python


"""Create plots for pinn1 results for the loop2d_nPuxuyuzBxByBz problem.

Create plots for pinn1 results for the loop2d_nPuxuyuzBxByBz problem.

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
    "clobber": common.DEFAULT_ARGUMENTS["clobber"],
    "debug": common.DEFAULT_ARGUMENTS["debug"],
    "image_format": "png",
    "min_epoch": -1,
    "verbose": common.DEFAULT_ARGUMENTS["verbose"],
    "results_path": None,
}

# Name of problem
PROBLEM_NAME = "loop2d_nPuxuyuzBxByBz"

# Name of directory to hold output plots
OUTPUT_DIR = "pinn1_plots"


def create_command_line_parser():
    """Create the command-line parser.

    Create the command-line parser.

    Parameters
    ----------
    None

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
        "--min_epoch", type=int,
        default=DEFAULT_ARGUMENTS["min_epoch"],
        help="Epoch for model to use (-1 for last) (default: %(default)s)"
    )
    parser.add_argument(
        "results_path",
        help="Path to directory containing results to plot."
    )
    return parser


def create_output_directory(clobber: bool = False):
    """Create the output directory for the plots.

    Create the output directory for the plots. The name of the output
    directory is "pinn1_plots".

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
    output_dir = "pinn1_plots"
    if os.path.isdir(output_dir) and clobber:
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    return output_dir


def make_pinn1_loss_plot(Lres: np.ndarray, Ldat: np.ndarray, L: np.ndarray,
                         **kwargs):
    """Make a plot of the pinn1 model loss history.

    Make a plot of the pinn1 model loss history.

    Parameters
    ----------
    Lres : np.ndarray, shape (n_epochs,)
        Values of residual loss for each epoch.
    Ldat : np.ndarray, shape (n_epochs,)
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
    # Extract optional keywords.
    title = kwargs.get("title", "")
    figsize = kwargs.get("figsize", None)

    # Create the figure and Axes.
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the data.
    ax.semilogy(Lres, label="$L_{res}$")
    ax.semilogy(Ldat, label="$L_{data}$")
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

    # Plot the fields.
    axp.quiver(X, Y, Bxp, Byp)
    axp.set_title("Predicted")
    axa.quiver(X, Y, Bxa, Bya)
    axa.set_title("Analytical")
    axe.quiver(X, Y, Bxe, Bye)
    axe.set_title("Error")

    # Set common plot options.
    for ax in axs:
        ax.grid()
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    # Decorate the plot.
    fig.suptitle(title)

    # Return the figure.
    return fig


def assemble_movie(movie_file: str, frame_pattern: str, frame_rate: int):
    """Assemble a movie from individual frames.

    Assemble a movie from individual frames.

    Parameters
    ----------
    movie_file : str
        Path to movie file.
    frame_pattern : str
        glob pattern for frame file names.
    frame_rate : int
        Frame rate (frames/second).

    Returns
    -------
    None

    Raises
    ------
    None
    """
    cmd = (
        f"ffmpeg -r {frame_rate} -i {frame_pattern} -vcodec mpeg4 "
        f"-pix_fmt yuv420p {movie_file}"
    )
    subprocess.run(cmd, shell=True, check=True)


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
    image_format = args["image_format"]
    min_epoch = args["min_epoch"]
    verbose = args["verbose"]
    results_path = args["results_path"]

    # ------------------------------------------------------------------------

    # Add the run results directory at the head of the module search path.
    sys.path.insert(0, results_path)

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

    # Create the plots in a memory buffer.
    mpl.use("Agg")

    # ------------------------------------------------------------------------

    # Load the data.

    # Loss (residual)
    if verbose:
        print("Loading residual loss data.")
    path = os.path.join(results_path, "L_res.dat")
    Lres = np.loadtxt(path)
    if debug:
        print(f"Lres = {Lres}")

    # Loss (data)
    if verbose:
        print("Loading data loss data.")
    path = os.path.join(results_path, "L_dat.dat")
    Ldat = np.loadtxt(path)
    if debug:
        print(f"Ldat = {Ldat}")

    # Loss (aggregate)
    if verbose:
        print("Loading loss data.")
    path = os.path.join(results_path, "L.dat")
    L = np.loadtxt(path)
    if debug:
        print(f"L = {L}")

    # Training data.
    path = os.path.join(results_path, "XY_data.dat")
    if verbose:
        print(f"Loading training data from {path}.")
    # column_names_data is a list of str, length p.n_var.
    # column_descriptions_data is a dict of dicts, length p.n_var.
    # XY_data.dat is a np.ndarray of float, shape (n_data, p.n_dim + p.n_var).
    column_names_data, column_descriptions_data, XY_data = (
        common.read_grid_file(path)
    )
    if debug:
        print(f"column_names_data = {column_names_data}")
        print(f"column_descriptions_data = {column_descriptions_data}")
        print(f"XY_data = {XY_data}")

    # Extract the independent and dependent variables.
    # Xd is np.ndarray, shape (n_data, p.n_dim).
    # Yd is np.ndarray, shape (n_data, p.n_var).
    Xd = XY_data[:, :p.n_dim]
    Yd = XY_data[:, p.n_dim:]
    if debug:
        print(f"Xd = {Xd}")
        print(f"Yd = {Yd}")

    # Count the data points.
    n_data = XY_data.shape[0]
    if debug:
        print(f"n_data = {n_data}")

    # Load the training points.
    path = os.path.join(results_path, "X_train.dat")
    if verbose:
        print(f"Loading training points from {path}.")
    column_names_train, column_descriptions_train, X_train = (
        common.read_grid_file(path)
    )
    if debug:
        print(f"column_names_train = {column_names_train}")
        print(f"column_descriptions_train = {column_descriptions_train}")
        print(f"X_train = {X_train}")

    # Count the training points.
    n_train = X_train.shape[0]
    if debug:
        print(f"n_train = {n_train}")

    # Extract grid counts.
    nt = column_descriptions_train["t"]["n"]
    nx = column_descriptions_train["x"]["n"]
    ny = column_descriptions_train["y"]["n"]
    nxy = nx*ny
    if debug:
        print(f"nt = {nt}")
        print(f"nx = {nx}")
        print(f"ny = {ny}")
        print(f"nxy = {nxy}")

    # Extract and reshape the x- and y-grid values from the first time step.
    # It is assumed to be the same for all other time steps.
    # NOTE: Extracted values of data must also use this reshape(ny, nx).T.
    Xg = X_train[:nxy, 1].reshape(ny, nx).T
    Yg = X_train[:nxy, 2].reshape(ny, nx).T
    if debug:
        print(f"Xg = {Xg}")
        print(f"Yg = {Yg}")

    # ------------------------------------------------------------------------

    # Load the trained models.

    # Find the epoch of the last trained model, or the epoch of the earliest
    # model >= min_epoch.
    last_epoch = common.find_last_epoch(results_path, min_epoch)
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

    # Trim the loss data to the last epoch if needed.
    if min_epoch != -1:
        if verbose:
            print(f"Trimming loss data to epoch {last_epoch}.")
        Lres = Lres[:last_epoch + 1, :]
        Ldat = Ldat[:last_epoch + 1, :]
        L = L[:last_epoch + 1, :]
        if debug:
            print(f"L = {L}")
            print(f"Lres = {Lres}")
            print(f"Ldat = {Ldat}")

    # ------------------------------------------------------------------------

    # terminal "s" -> list of np.ndarray or tf.Variable/Tensor

    # Compute the predicted solutions and error at the data points.
    if verbose:
        print("Computing predicted values and error for data points.")
    Ydps = [model(Xd).numpy() for model in models]
    if debug:
        print(f"Ydps = {Ydps}")
    Ydp = np.hstack(Ydps)
    Yde = Ydp - Yd
    if debug:
        print(f"Ydp = {Ydp}")
        print(f"Yde = {Yde}")

    # Compute the predicted and analytical solutions and derivatives, and
    # errors at the training points.
    if verbose:
        print("Computing predicted and analytical solutions and derivatives, "
              "and error for training points.")
    Xt_tf = tf.Variable(X_train)
    with tf.GradientTape(persistent=True) as tape1:
        Ytps = [model(Xt_tf) for model in models]
    dYtp_dXts = [tape1.gradient(Y, Xt_tf) for Y in Ytps]
    if debug:
        print(f"Ytps = {Ytps}")
        print(f"dYtp_dXts = {dYtp_dXts}")
    Ytps = [Y.numpy() for Y in Ytps]
    Ytp = np.hstack(Ytps)
    dYtp_dXts = [dY_dX.numpy() for dY_dX in dYtp_dXts]
    if debug:
        print(f"Ytps = {Ytps}")
        print(f"Ytp = {Ytp}")
        print(f"dYtp_dXts = {dYtp_dXts}")
    Yta = [f(X_train) for f in p.Y_analytical]
    Yta = np.hstack(Yta)
    Yte = Ytp - Yta
    if debug:
        print(f"Ytp = {Ytp}")
        print(f"Yta = {Yta}")
        print(f"Yte = {Yte}")
    dBx_dxa = p.dBx_dx_analytical(X_train)
    dBy_dya = p.dBy_dy_analytical(X_train)
    if debug:
        print(f"dBx_dxa = {dBx_dxa}")
        print(f"dBy_dya = {dBy_dya}")

    # Compute the predicted and analytical derivatives and magnetic divergence,
    # and error, at the training points.
    if verbose:
        print("Computing predicted and analytical magnetic divergence, and "
              "error at training points.")
    dBx_dxp = dYtp_dXts[p.iBx][:, p.ix]
    dBx_dxe = dBx_dxp - dBx_dxa
    dBy_dyp = dYtp_dXts[p.iBy][:, p.iy]
    dBy_dye = dBy_dyp - dBy_dya
    divBp = dBx_dxp**2 + dBy_dyp**2
    divBa = dBx_dxa**2 + dBy_dya**2
    divBe = divBp - divBa
    if debug:
        print(f"dBx_dxp = {dBx_dxp}")
        print(f"dBx_dxe = {dBx_dxe}")
        print(f"dBy_dyp = {dBy_dyp}")
        print(f"dBy_dye = {dBy_dye}")
        print(f"divBp = {divBp}")
        print(f"divBa = {divBa}")
        print(f"divBe = {divBe}")

    # Compute the predicted and analytical magnetic pressure and error.
    if verbose:
        print("Computing predicted and analytical magnetic pressure, and "
              "error at training points.")
    PBp = Ytp[:, p.iBx]**2 + Ytp[:, p.iBy]**2
    PBa = Yta[:, p.iBx]**2 + Yta[:, p.iBy]**2
    PBe = PBp - PBa
    if debug:
        print(f"PBp = {PBp}")
        print(f"PBa = {PBa}")
        print(f"PBe = {PBe}")

    # ------------------------------------------------------------------------

    # Create the plots in a memory buffer.
    mpl.use("Agg")

    # ------------------------------------------------------------------------

    # Plot loss histories.

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
        fig = make_pinn1_loss_plot(
            Lres[:, iv], Ldat[:, iv], L[:, iv], title=title
        )

        # Save the plot to a file.
        path = os.path.join(output_dir, f"L_{variable_name}.{image_format}")
        if verbose:
            print(f"Saving {path}.")
        plt.savefig(path)

        # Close the figure.
        plt.close(fig)

        # End of variable loop.

    # Create the figure.
    if verbose:
        print("Creating aggregate loss plot.")
    title = "Total residual, data, and weighted loss"
    fig = make_pinn1_loss_plot(
        Lres[:, -1], Ldat[:, -1], L[:, -1], title=title
    )

    # Save the plot to a file.
    path = os.path.join(output_dir, f"L.{image_format}")
    if verbose:
        print(f"Saving {path}.")
    plt.savefig(path)

    # Close the figure.
    plt.close(fig)

    # ------------------------------------------------------------------------

    # Compute the number of frames in each movie as the number of time values.
    # This code assumes the same xy points used at each time.
    n_frames = nt

    # Specify the frame rate in frames/second.
    frame_rate = 2

    # ------------------------------------------------------------------------

    # Make a movie of the predicted and analytical solutions, and the error,
    # for each model.

    # Constant plot parameters
    figsize = (12, 5)
    xlabel = p.independent_variable_labels[p.ix]
    ylabel = p.independent_variable_labels[p.iy]

    # Make a plot for each model.
    for iv in range(p.n_var):

        # Extract the variable name and label.
        variable_name = p.dependent_variable_names[iv]
        variable_label = p.dependent_variable_labels[iv]

        if verbose:
            print("Creating predicted/analytical/error movie for "
                  f"{variable_name}.")

        # Create the directory for the frames for this variable.
        frame_dir = os.path.join(output_dir, f"frames_{variable_name}")
        os.mkdir(frame_dir)

        # Create the frames.
        if verbose:
            print(f"Creating frames for {variable_name}.")
        for i_frame in range(n_frames):
            if verbose:
                print(f"Creating {variable_name} frame {i_frame}.")

            # Extract the time of the frame.
            t = X_train[i_frame*nxy, 0]

    #         # Extract the data for the frame.
            i1 = i_frame*nxy
            i2 = (i_frame + 1)*nxy
            # NOTE: Needs same reshape().T used by grid points,.
            X = X_train[i1:i2, p.ix].reshape(ny, nx).T
            Y = X_train[i1:i2, p.iy].reshape(ny, nx).T
            Zp = Ytp[i1:i2, iv].reshape(ny, nx).T
            Za = Yta[i1:i2, iv].reshape(ny, nx).T
            Ze = Yte[i1:i2, iv].reshape(ny, nx).T

            # Compute the frame title.
            title = f"{variable_label} at t = {t:0.3E}"
            fig = make_PAE_plot(
                Zp, Za, Ze, X, Y,
                title=title, xlabel=xlabel, ylabel=ylabel, figsize=figsize
            )

            # Save the plot to a file.
            path = os.path.join(
                frame_dir, f"{variable_name}-{i_frame:06d}.{image_format}")
            if verbose:
                print(f"Saving {path}.")
            plt.savefig(path)

            # Close the figure.
            plt.close(fig)

            # End of frame loop.

        # Assemble the frames into a movie.
        if verbose:
            print(f"Assembling frames for {variable_name}.")
        frame_pattern = os.path.join(
            frame_dir, f"{variable_name}-%06d.{image_format}"
        )
        movie_file = os.path.join(output_dir, f"{variable_name}.mp4")
        assemble_movie(movie_file, frame_pattern, frame_rate)

        # End of variable loop.

    # ------------------------------------------------------------------------

    # Make a movie of the predicted and analytical magnetic field vectors, and
    # the error.
    if verbose:
        print("Creating predicted/analytical/error movie for xy-magnetic "
              "field.")

    # Create the directory for the frames.
    frame_dir = os.path.join(output_dir, "frames_BxBy")
    os.mkdir(frame_dir)

    # Create the frames.
    if verbose:
        print("Creating frames for magnetic field vector movie.")
    for i_frame in range(n_frames):
        if verbose:
            print(f"Creating BxBy frame {i_frame}.")

        # Extract the time of the frame.
        t = X_train[i_frame*nxy, p.it]

        # Extract the data for the frame.
        i1 = i_frame*nxy
        i2 = (i_frame + 1)*nxy
        # NOTE: Needs same reshape().T used by grid points,.
        Bxp = Ytp[i1:i2, p.iBx].reshape(ny, nx).T
        Byp = Ytp[i1:i2, p.iBy].reshape(ny, nx).T
        Bxa = Yta[i1:i2, p.iBx].reshape(ny, nx).T
        Bya = Yta[i1:i2, p.iBy].reshape(ny, nx).T
        Bxe = Bxp - Bxa
        Bye = Byp - Bya

        # Compute the frame title.
        title = f"Magnetic field at t = {t:0.3E}"

        # Plot the magnetic field vectors.
        fig = make_PAE_B_plot(Bxp, Byp, Bxa, Bya, Bxe, Bye, Xg, Yg,
                              title=title, figsize=figsize)

        # Save the plot to a file.
        path = os.path.join(
            frame_dir, f"BxBy-{i_frame:06d}.{image_format}")
        if verbose:
            print(f"Saving {path}.")
        plt.savefig(path)

        # Close the figure.
        plt.close(fig)

        # End of frame loop.

    # Assemble the frames into a movie.
    if verbose:
        print("Assembling frames for BxBy.")
    frame_pattern = os.path.join(frame_dir, f"BxBy-%06d.{image_format}")
    movie_file = os.path.join(output_dir, "BxBy.mp4")
    assemble_movie(movie_file, frame_pattern, frame_rate)

    # ------------------------------------------------------------------------

    # Make a movie of the predicted and analytical magnetic divergence, and
    # the error.
    if verbose:
        print("Creating predicted/analytical/error movie for magnetic "
              "divergence.")

    # Create the directory for the frames.
    frame_dir = os.path.join(output_dir, "frames_divB")
    os.mkdir(frame_dir)

    # Create the frames.
    if verbose:
        print("Creating frames for magnetic divergence movie.")
    for i_frame in range(n_frames):
        if verbose:
            print(f"Creating divB frame {i_frame}.")

        # Extract the time of the frame.
        t = X_train[i_frame*nxy, p.it]

        # Extract the data for the frame.
        i1 = i_frame*nxy
        i2 = (i_frame + 1)*nxy
        # NOTE: Needs same reshape().T used by grid points,.
        Zp = divBp[i1:i2].reshape(ny, nx).T
        Za = divBa[i1:i2].reshape(ny, nx).T
        Ze = divBe[i1:i2].reshape(ny, nx).T

        # Compute the frame title.
        title = f"Magnetic divergence at t = {t:0.3E}"

        # Plot the magnetic field vectors.
        fig = make_PAE_plot(Zp, Za, Ze, Xg, Yg, title=title, figsize=figsize)

        # Save the plot to a file.
        path = os.path.join(
            frame_dir, f"divB-{i_frame:06d}.{image_format}")
        if verbose:
            print(f"Saving {path}.")
        plt.savefig(path)

        # Close the figure.
        plt.close(fig)

        # End of frame loop.

    # Assemble the frames into a movie.
    if verbose:
        print("Assembling frames for divB.")
    frame_pattern = os.path.join(frame_dir, f"divB-%06d.{image_format}")
    movie_file = os.path.join(output_dir, "divB.mp4")
    assemble_movie(movie_file, frame_pattern, frame_rate)

    # ------------------------------------------------------------------------

    # Make a movie of the predicted and analytical magnetic pressure, and
    # the error.
    if verbose:
        print("Creating predicted/analytical/error movie for magnetic "
              "pressure.")

    # Create the directory for the frames.
    frame_dir = os.path.join(output_dir, "frames_PB")
    os.mkdir(frame_dir)

    # Create the frames.
    if verbose:
        print("Creating frames for magnetic pressure movie.")
    for i_frame in range(n_frames):
        if verbose:
            print(f"Creating PB frame {i_frame}.")

        # Extract the time of the frame.
        t = X_train[i_frame*nxy, p.it]

        # Extract the data for the frame.
        i1 = i_frame*nxy
        i2 = (i_frame + 1)*nxy
        # NOTE: Needs same reshape().T used by grid points,.
        Zp = PBp[i1:i2].reshape(ny, nx).T
        Za = PBa[i1:i2].reshape(ny, nx).T
        Ze = PBe[i1:i2].reshape(ny, nx).T

        # Compute the frame title.
        title = f"Magnetic pressure at t = {t:0.3E}"

        # Plot the magnetic field vectors.
        fig = make_PAE_plot(Zp, Za, Ze, Xg, Yg, title=title, figsize=figsize)

        # Save the plot to a file.
        path = os.path.join(
            frame_dir, f"PB-{i_frame:06d}.{image_format}")
        if verbose:
            print(f"Saving {path}.")
        plt.savefig(path)

        # Close the figure.
        plt.close(fig)

        # End of frame loop.

    # Assemble the frames into a movie.
    if verbose:
        print("Assembling frames for PB.")
    frame_pattern = os.path.join(frame_dir, f"PB-%06d.{image_format}")
    movie_file = os.path.join(output_dir, "PB.mp4")
    assemble_movie(movie_file, frame_pattern, frame_rate)


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

    # Pass the command-line arguments to the main function as a dict.
    return_code = pinn1_plots(args)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
