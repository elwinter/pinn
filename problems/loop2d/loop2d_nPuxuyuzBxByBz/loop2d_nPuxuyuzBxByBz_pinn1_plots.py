#!/usr/bin/env python

"""Create plots for pinn1 results for loop2d_nPuxuyuzBxByBz problem.

Create plots for pinn1 results for loop2d_nPuxuyuzBxByBz problem.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

# Import standard modules.
import argparse
import copy
import importlib
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
import pinn.common
# import pinn.standard_plots


# Program constants

# Program description
DESCRIPTION = (
    "Create plots for pinn1 results for loop2d_nPuxuyuzBxByBz problem."
)

# Default values for command-line arguments.
DEFAULT_ARGUMENTS = {
    "clobber": False,
    "debug": False,
    "epoch": -1,
    "verbose": False,
}

# Name of problem
PROBLEM_NAME = "loop2d_nPuxuyuzBxByBz"

# Name of directory to hold output plots
OUTPUT_DIR = "pinn1_plots"

# Movie parameters
FRAME_RATE = "10"  # Frames per second
FRAME_SIZE = "1920x1080"
VIDEO_CODEC = "libx264"
CONSTANT_RATE_FACTOR = "25"
PIXEL_FORMAT = "yuv420p"


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
    parser = argparse.ArgumentParser(DESCRIPTION)
    parser.add_argument(
        "--clobber",
        default=DEFAULT_ARGUMENTS["clobber"],
        action="store_true",
        help="Overwrite existing plots (default: %(default)s)."
    )
    parser.add_argument(
        "--debug", "-d",
        default=DEFAULT_ARGUMENTS["debug"],
        action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=DEFAULT_ARGUMENTS["epoch"],
        help="Model epoch to use (default: %(default)s)."
    )
    parser.add_argument(
        "--verbose", "-v",
        default=DEFAULT_ARGUMENTS["verbose"],
        action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "results_path",
        help="Path to directory containing results to plot."
    )
    return parser


def create_loss_plot(L_res: np.ndarray, L_dat: np.ndarray,
                     L: np.ndarray) -> mpl.pyplot.Figure:
    """Create a plot of residual, model, and weighted loss.

    Create a plot of residual, model, and weighted loss.

    Parameters
    ----------
    L_res : np.ndarray, shape (n_epochs,)
        Residual loss values
    L_dat : np.ndarray, shape (n_epochs,)
        Data loss values
    L : np.ndarray, shape (n_epochs,)
        Weighted loss values

    Returns
    -------
    fig : mpl.pyplot.Figure
        Figure object for current plot

    Raises
    ------
    None
    """
    # Create the figure.
    fig = plt.figure()

    # Plot the residual, data, and weighted losses.
    plt.semilogy(L_res, label="$L_{res}$")
    plt.semilogy(L_dat, label="$L_{dat}$")
    plt.semilogy(L, label="$L$")

    # Decorate the plot.
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Residual, data, and weighted loss")
    plt.grid()

    # Return the figure.
    return fig


def create_PAE_plot(X: np.ndarray, Y: np.ndarray,
                    P: np.ndarray, A: np.ndarray,
                    E: np.ndarray) -> mpl.pyplot.Figure:
    """Create a plot of predicted, analytical, and error values.

    Create a plot of predicted, analytical, and error values.

    Parameters
    ----------
    X : np.ndarray, shape (ny, nx)
        X values
    Y : np.ndarray, shape (ny, nx)
        Y values
    P : np.ndarray, shape (ny, nx)
        Predicted values
    A : np.ndarray, shape (ny, nx)
        Analytical values
    E : np.ndarray, shape (ny, nx)
        Error values

    Returns
    -------
    fig : mpl.pyplot.Figure
        Figure object for current plot

    Raises
    ------
    None
    """
    # Create the figure.
    fig, axs = plt.subplots(
        nrows=1, ncols=3, sharey=True,
        figsize=[18.0, 6.0]
        )

    # Predicted
    pcmp = axs[0].pcolormesh(X, Y, P)
    axs[0].set_aspect("equal")
    axs[0].set_title("Predicted")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].grid(True)
    fig.colorbar(pcmp, ax=axs[0], orientation="horizontal")

    # Analytical
    pcma = axs[1].pcolormesh(X, Y, A)
    axs[1].set_aspect("equal")
    axs[1].set_title("Analytical")
    axs[1].set_xlabel("x")
    axs[1].grid(True)
    fig.colorbar(pcma, ax=axs[1], orientation="horizontal")

    # Error
    pcme = axs[2].pcolormesh(X, Y, E)
    axs[2].set_aspect("equal")
    axs[2].set_title("Error")
    axs[2].set_xlabel("x")
    axs[2].grid(True)
    fig.colorbar(pcme, ax=axs[2], orientation="horizontal")

    # Decorate the figure.
    fig.suptitle("Predicted, analytical, and error")

    # Return the figure.
    return fig


def create_PA_BxBy_plot(X: np.ndarray, Y: np.ndarray,
                        Px: np.ndarray, Py: np.ndarray,
                        Ax: np.ndarray, Ay: np.ndarray) -> mpl.pyplot.Figure:
    """Create a plot of predicted and analytical magnetic field.

    Create a plot of predicted and analytical magnetic field.

    Parameters
    ----------
    X : np.ndarray, shape (ny, nx)
        X values
    Y : np.ndarray, shape (ny, nx)
        Y values
    Px : np.ndarray, shape (ny, nx)
        Predicted Bx values
    Py : np.ndarray, shape (ny, nx)
        Predicted By values
    Ax : np.ndarray, shape (ny, nx)
        Analytical Bx values
    Ay : np.ndarray, shape (ny, nx)
        Analytical By values

    Returns
    -------
    fig : mpl.pyplot.Figure
        Figure object for current plot

    Raises
    ------
    None
    """
    # Create the figure.
    fig, axs = plt.subplots(
        nrows=1, ncols=2, sharey=True,
        figsize=[12.0, 6.0]
        )

    # Predicted
    axs[0].quiver(X, Y, Px, Py)
    axs[0].set_title("Predicted")
    axs[0].set_aspect("equal")

    # Analytical
    axs[1].quiver(X, Y, Ax, Ay)
    axs[1].set_title("Analytical")
    axs[1].set_aspect("equal")

    # Decorate the figure.
    fig.suptitle("Predicted and analytical magnetic field")

    # Return the figure.
    return fig


def assemble_movie(frame_pattern: str, movie_file: str) -> None:
    """Assemble a movie from individual frames.

    Assemble a movie from individual frames.

    Parameters
    ----------
    frame_pattern : str
        Glob pattern for frame files.
    movie_file : str
        Path to movie file to create.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Assemble the frames into a movie.
    args = [
        "ffmpeg", "-r", FRAME_RATE, "-s", FRAME_SIZE,
        "-i", frame_pattern, "-vcodec", VIDEO_CODEC,
        "-crf", CONSTANT_RATE_FACTOR, "-pix_fmt", PIXEL_FORMAT,
        movie_file
    ]
    subprocess.run(args, check=True)


def pinn1_plots(**kwargs) -> int:
    """Create pinn1 plots for the loop2d_nPuxuyuzBxByBz problem.

    Create pinn1 plots for the loop2d_nPuxuyuzBxByBz problem.

    Parameters
    ----------
    kwargs : dict
        Dictionary of keyword arguments.

    Returns
    -------
    int 0 on success, otherwise Exception is raised by called code.

    Raises
    ------
    None
    """
    # Set defaults for command-line options, then update with values passed
    # from the caller.
    args = copy.deepcopy(DEFAULT_ARGUMENTS)
    args.update(kwargs)

    # Local convenience variables.
    debug = args["debug"]
    verbose = args["verbose"]
    results_path = args["results_path"]
    if debug:
        print(f"debug = {debug}")
        print(f"verbose = {verbose}")
        print(f"results_path = {results_path}")

    # ------------------------------------------------------------------------

    # Add the run results directory to the module search path.
    sys.path.append(results_path)

    # Import the problem definition from the run results directory.
    p = importlib.import_module(PROBLEM_NAME)

    # Compute the path to the output directory to hold the plots. Then create
    # it if needed. Exception will be raised if directory exists and clobber
    # is not set.
    output_path = OUTPUT_DIR
    if os.path.isdir(output_path):
        if args["clobber"]:
            if verbose:
                print(f"Deleting existing output directory {output_path}.")
            shutil.rmtree(output_path)
    os.mkdir(output_path)

    # ------------------------------------------------------------------------

    # Create the plots in a memory buffer.
    mpl.use("Agg")

    # ------------------------------------------------------------------------

    # Plot the total residual, data, and weighted loss histories.

    # Load the data.
    if verbose:
        print("Loading aggregate loss data.")
    path = os.path.join(results_path, "L_res.dat")
    L_res = np.loadtxt(path)
    path = os.path.join(results_path, "L_data.dat")
    L_dat = np.loadtxt(path)
    path = os.path.join(results_path, "L.dat")
    L = np.loadtxt(path)

    # Create the plot.
    if verbose:
        print("Creating aggregate loss plot.")
    fig = create_loss_plot(L_res, L_dat, L)
    ax = fig.get_axes()[0]
    ax.set_title("Aggregate residual, data, and weighted loss")

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "L.png")
    if verbose:
        print(f"Saving {path}.")
    fig.savefig(path)

    # ------------------------------------------------------------------------

    # Plot the per-model residual, data, and weighted loss histories.

    # Plot for each model.
    for iv in range(p.n_var):
        variable_name = p.dependent_variable_names[iv]
        if verbose:
            print(f"Creating loss plot for {variable_name}.")

        # Load the data.
        path = os.path.join(results_path, f"L_res_{variable_name}.dat")
        L_res = np.loadtxt(path)
        path = os.path.join(results_path, f"L_data_{variable_name}.dat")
        L_dat = np.loadtxt(path)
        path = os.path.join(results_path, f"L_{variable_name}.dat")
        L = np.loadtxt(path)

        # Create the plot.
        variable_label = p.dependent_variable_labels[iv]
        fig = create_loss_plot(L_res, L_dat, L)
        ax = fig.get_axes()[0]
        ax.set_title(f"{variable_label} residual, data, and weighted loss")

        # Save the plot to a PNG file.
        path = os.path.join(output_path, f"L_{variable_name}.png")
        fig.savefig(path)
        plt.close(fig)

    # ------------------------------------------------------------------------

    # Load the training points and description.
    path = os.path.join(results_path, "X_train.dat")
    X_train = np.loadtxt(path)
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()  # Skip 1st line - contains "# GRID"
        line = f.readline()  # Grid description on this line
        line = line[2:]
        fields = line.split(" ")
        tmin = float(fields[0])
        tmax = float(fields[1])
        nt = int(fields[2])
        xmin = float(fields[3])
        xmax = float(fields[4])
        nx = int(fields[5])
        ymin = float(fields[6])
        ymax = float(fields[7])
        ny = int(fields[8])
    if debug:
        print(f"(tmin, tmax, nt) = ({tmin}, {tmax}, {nt})")
        print(f"(xmin, xmax, nx) = ({xmin}, {xmax}, {nx})")
        print(f"(ymin, ymax, ny) = ({ymin}, {ymax}, {ny})")

    # Determine the epoch of the trained model to use.
    if args["epoch"] == -1:
        epoch = pinn.common.find_last_epoch(results_path)
    else:
        epoch = args["epoch"]

    # Load the trained model for each variable.
    models = []
    for variable_name in p.dependent_variable_names:
        path = os.path.join(results_path, "models", f"{epoch:06d}",
                            f"model_{variable_name}")
        model = tf.keras.models.load_model(path)
        models.append(model)

    # ------------------------------------------------------------------------

    # Plot the predicted, analytical, and error values for each model as a
    # function of time, at the training points.

    # Plot for each model.
    for iv in range(p.n_var):
        variable_name = p.dependent_variable_names[iv]
        variable_label = p.dependent_variable_labels[iv]
        if verbose:
            print(f"Creating training point PAE plots for {variable_name}.")

        # Create a directory for the PAE plots for this variable.
        pae_path = os.path.join(output_path, f"PAE_{variable_name}")
        print(f"pae_path = {pae_path}")
        os.mkdir(pae_path)

        # Compute the PAE values.
        predicted = models[iv](X_train).numpy().reshape(nt, nx, ny)
        analytical = p.analytical_solutions[iv](
            X_train[:, p.it], X_train[:, p.ix], X_train[:, p.iy]
        ).reshape(nt, nx, ny)
        error = predicted - analytical

        # Plot for each training grid time.
        for it in range(nt):

            # Compute the starting and ending index for this time.
            i0 = it*nx*ny
            i1 = i0 + nx*ny

            # Fetch the frame time.
            t = X_train[i0, p.it]

            # Extract the X and Y values for this time.
            X = X_train[i0:i1, p.ix].reshape(nx, ny).T
            Y = X_train[i0:i1, p.iy].reshape(nx, ny).T

            # To get the proper orientation, reshape, transpose.
            P = predicted[it, :].T
            A = analytical[it, :].T
            E = error[it, :].T

            # Create the plot.
            fig = create_PAE_plot(X, Y, P, A, E)
            fig.suptitle(f"{variable_label}, t = {t:.2E} predicted, "
                         "analytical, and error")

            # Save the plot to a PNG file.
            path = os.path.join(pae_path, f"PAE_{variable_name}_{it:04d}.png")
            fig.savefig(path)
            plt.close(fig)

        # Assemble the frames into a movie.
        frame_pattern = os.path.join(pae_path, f"PAE_{variable_name}_%04d.png")
        movie_file = os.path.join(pae_path, f"PAE_{variable_name}.mp4")
        assemble_movie(frame_pattern, movie_file)

    # ------------------------------------------------------------------------

    # Make a movie of the magnetic field vectors.

    if verbose:
        print("Creating movie for magnetic field.")
    pa_path = os.path.join(output_path, "PA_BxBy")
    os.mkdir(pa_path)

    # Compute the predicted and analytical magnetic field components.
    Bxp = models[p.iBx](X_train).numpy().reshape(nt, nx, ny)
    Byp = models[p.iBy](X_train).numpy().reshape(nt, nx, ny)
    Bxa = p.analytical_solutions[p.iBx](
            X_train[:, p.it], X_train[:, p.ix], X_train[:, p.iy]
        ).reshape(nt, nx, ny)
    Bya = p.analytical_solutions[p.iBy](
            X_train[:, p.it], X_train[:, p.ix], X_train[:, p.iy]
        ).reshape(nt, nx, ny)

    # Plot the field at each time.
    for it in range(nt):

        # Compute the starting and ending index for this time.
        i0 = it*nx*ny
        i1 = i0 + nx*ny

        # Fetch the frame time.
        t = X_train[i0, p.it]

        # Extract the X and Y values for this time.
        X = X_train[i0:i1, p.ix].reshape(nx, ny).T
        Y = X_train[i0:i1, p.iy].reshape(nx, ny).T

        # To get the proper orientation, reshape, transpose.
        Px = Bxp[it, :].T
        Py = Byp[it, :].T
        Ax = Bxa[it, :].T
        Ay = Bya[it, :].T

        # Create the plot.
        fig = create_PA_BxBy_plot(X, Y, Px, Py, Ax, Ay)
        fig.suptitle(f"Magnetic field, t = {t:.2E} predicted, analytical")

        # Save the plot to a PNG file.
        path = os.path.join(pa_path, f"PA_BxBy_{it:04d}.png")
        fig.savefig(path)
        plt.close(fig)

    # Assemble the frames into a movie.
    frame_pattern = os.path.join(pa_path, "PA_BxBy_%04d.png")
    movie_file = os.path.join(pa_path, "PA_BxBy.mp4")
    assemble_movie(frame_pattern, movie_file)

    # ------------------------------------------------------------------------

    # Make a movie of the magnetic field intensity.

    if verbose:
        print("Creating movie for magnetic field intensity.")
    pae_path = os.path.join(output_path, "PAE_B")
    os.mkdir(pae_path)

    # Compute the predicted and analytical magnetic field intensity, and error.
    Bp = np.sqrt(Bxp**2 + Byp**2)
    Ba = np.sqrt(Bxa**2 + Bya**2)
    Be = Bp - Ba

    # Plot the field at each time.
    for it in range(nt):

        # Compute the starting and ending index for this time.
        i0 = it*nx*ny
        i1 = i0 + nx*ny

        # Fetch the frame time.
        t = X_train[i0, p.it]

        # Extract the X and Y values for this time.
        X = X_train[i0:i1, p.ix].reshape(nx, ny).T
        Y = X_train[i0:i1, p.iy].reshape(nx, ny).T

        # To get the proper orientation, reshape, transpose.
        P = Bp[it, :].T
        A = Ba[it, :].T
        E = Be[it, :].T

        # Create the plot.
        fig = create_PAE_plot(X, Y, P, A, E)
        fig.suptitle(f"Magnetic field intensity, t = {t:.2E} predicted, "
                     "analytical, and error")

        # Save the plot to a PNG file.
        path = os.path.join(pae_path, f"PAE_B_{it:04d}.png")
        fig.savefig(path)
        plt.close(fig)

    # Assemble the frames into a movie.
    frame_pattern = os.path.join(pae_path, "PAE_B_%04d.png")
    movie_file = os.path.join(pae_path, "PAE_B.mp4")
    assemble_movie(frame_pattern, movie_file)

    # ------------------------------------------------------------------------

    # Make a movie of the magnetic energy.

    if verbose:
        print("Creating movie for magnetic field energy.")
    pae_path = os.path.join(output_path, "PAE_Eb")
    os.mkdir(pae_path)

    # Compute the predicted and analytical magnetic field energy, and error.
    Ebp = Bxp**2 + Byp**2
    Eba = Bxa**2 + Bya**2
    Ebe = Ebp - Eba

    # Plot the field at each time.
    for it in range(nt):

        # Compute the starting and ending index for this time.
        i0 = it*nx*ny
        i1 = i0 + nx*ny

        # Fetch the frame time.
        t = X_train[i0, p.it]

        # Extract the X and Y values for this time.
        X = X_train[i0:i1, p.ix].reshape(nx, ny).T
        Y = X_train[i0:i1, p.iy].reshape(nx, ny).T

        # To get the proper orientation, reshape, transpose.
        P = Ebp[it, :].T
        A = Eba[it, :].T
        E = Ebe[it, :].T

        # Create the plot.
        fig = create_PAE_plot(X, Y, P, A, E)
        fig.suptitle(f"Magnetic field energy, t = {t:.2E} predicted, "
                     "analytical, and error")

        # Save the plot to a PNG file.
        path = os.path.join(pae_path, f"PAE_Eb_{it:04d}.png")
        fig.savefig(path)
        plt.close(fig)

    # Assemble the frames into a movie.
    frame_pattern = os.path.join(pae_path, "PAE_Eb_%04d.png")
    movie_file = os.path.join(pae_path, "PAE_Eb.mp4")
    assemble_movie(frame_pattern, movie_file)

    # ------------------------------------------------------------------------

    # Make a movie of the magnetic field divergence.

    if verbose:
        print("Creating movie for magnetic field divergence.")
    pae_path = os.path.join(output_path, "PAE_divB")
    os.mkdir(pae_path)

    # Compute the predicted and analytical magnetic field divergence, and
    # error.
    txyv = tf.Variable(X_train)
    with tf.GradientTape(persistent=True) as tape1:
        Bxp = models[p.iBx](txyv)
        Byp = models[p.iBy](txyv)
    dBxp_dx = tape1.gradient(Bxp, txyv)[:, p.ix].numpy()
    dByp_dy = tape1.gradient(Byp, txyv)[:, p.iy].numpy()
    divBp = dBxp_dx + dByp_dy
    divBa = np.zeros(divBp.shape)
    divBe = divBp - divBa

    # Plot the field at each time.
    for it in range(nt):

        # Compute the starting and ending index for this time.
        i0 = it*nx*ny
        i1 = i0 + nx*ny

        # Fetch the frame time.
        t = X_train[i0, p.it]

        # Extract the X and Y values for this time.
        X = X_train[i0:i1, p.ix].reshape(nx, ny).T
        Y = X_train[i0:i1, p.iy].reshape(nx, ny).T

        # To get the proper orientation, reshape, transpose.
        P = divBp[i0:i1].reshape(nx, ny).T
        A = divBa[i0:i1].reshape(nx, ny).T
        E = divBe[i0:i1].reshape(nx, ny).T

        # Create the plot.
        fig = create_PAE_plot(X, Y, P, A, E)
        fig.suptitle(f"Magnetic field divergence, t = {t:.2E} predicted, "
                     "analytical, and error")

        # Save the plot to a PNG file.
        path = os.path.join(pae_path, f"PAE_divB_{it:04d}.png")
        fig.savefig(path)
        plt.close(fig)

    # Assemble the frames into a movie.
    frame_pattern = os.path.join(pae_path, "PAE_divB_%04d.png")
    movie_file = os.path.join(pae_path, "PAE_divB.mp4")
    assemble_movie(frame_pattern, movie_file)

    # ------------------------------------------------------------------------

    # Return normally.
    return 0


def main() -> None:
    """Driver for command-line version of code.

    This is the main function for the command-line version of this file. It
    processes the command-line arguments and calls the primary script code.

    Parameters
    ----------
    None

    Returns
    -------
    return_code : int
        Return code from primary script code.

    Raises
    ------
    None
    """
    # Create the command-line parser.
    parser = create_command_line_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}")

    # Convert the arguments from Namespace to dict.
    args = vars(args)

    # Call the main program code.
    return_code = pinn1_plots(**args)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
