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
    "usetex": False,
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
        "--usetex",
        default=DEFAULT_ARGUMENTS["usetex"],
        action="store_true",
        help="Use LaTeX in plots (default: %(default)s)"
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


def create_rms_error_plot(t: np.ndarray, rms: np.ndarray
                          ) -> mpl.pyplot.Figure:
    """Create a plot of RMS error over time.

    Create a plot of RMS error over time.

    Parameters
    ----------
    t : np.ndarray, shape (nt,)
        Time values
    rms : np.ndarray, shape (nt,)
        RMS error values

    Returns
    -------
    fig : mpl.pyplot.Figure
        Figure object for current plot

    Raises
    ------
    None
    """
    # Create the figure.
    fig, ax = plt.subplots()

    # Plot the RMS error over time.
    ax.plot(t, rms)

    # Decorate the figure.
    ax.set_title("RMS Error")
    ax.set_xlabel("t")
    ax.set_ylabel("RMS error")

    # Return the figure.
    return fig


def create_total_magnetic_energy_plot(
        t: np.ndarray, Ebtot: np.ndarray) -> mpl.pyplot.Figure:
    """Create a plot of total magnetic energy over time.

    Create a plot of total magnetic energy over time.

    Parameters
    ----------
    t : np.ndarray, shape (nt,)
        Time values
    Ebtot : np.ndarray, shape (nt,)
        Total magnetic energy values

    Returns
    -------
    fig : mpl.pyplot.Figure
        Figure object for current plot

    Raises
    ------
    None
    """
    # Create the figure.
    fig, ax = plt.subplots()

    # Plot the total magnetic energy over time.
    ax.plot(t, Ebtot)

    # Decorate the figure.
    ax.set_title("Total Magnetic Energy")
    ax.set_xlabel("t")
    ax.set_ylabel("$E_{btot}$")

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
    subprocess.run(args, check=True, capture_output=True)


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

    # Load all data.

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

    # Load the aggregate loss histories.
    path = os.path.join(results_path, "L_res.dat")
    L_res = np.loadtxt(path)
    path = os.path.join(results_path, "L_data.dat")
    L_dat = np.loadtxt(path)
    path = os.path.join(results_path, "L.dat")
    L = np.loadtxt(path)

    # Load the per-model residual, data, and weighted loss histories.
    Lm_res = []
    Lm_dat = []
    Lm = []
    for iv in range(p.n_var):
        variable_name = p.dependent_variable_names[iv]
        path = os.path.join(results_path, f"L_res_{variable_name}.dat")
        Lm_res.append(np.loadtxt(path))
        path = os.path.join(results_path, f"L_data_{variable_name}.dat")
        Lm_dat.append(np.loadtxt(path))
        path = os.path.join(results_path, f"L_{variable_name}.dat")
        Lm.append(np.loadtxt(path))

    # ------------------------------------------------------------------------

    # Compute derived values.

    # Extract the T, X, and Y values for the training points.
    T = X_train[:, p.it].reshape(nt, nx, ny)
    X = X_train[:, p.ix].reshape(nt, nx, ny)
    Y = X_train[:, p.iy].reshape(nt, nx, ny)

    # Compute predicted, analytical, and error values for each model at each
    # training point. All are shape (nt, nx, ny).
    predicted = {}
    analytical = {}
    error = {}
    for iv in range(p.n_var):
        variable_name = p.dependent_variable_names[iv]
        _p = models[iv](X_train).numpy().reshape(nt, nx, ny)
        predicted[variable_name] = _p
        _a = p.analytical_solutions[iv](T, X, Y)
        analytical[variable_name] = _a
        _e = _p - _a
        error[variable_name] = _e

    # Magnetic energy
    _p = predicted["Bx"]**2 + predicted["By"]**2
    predicted["Eb"] = _p
    _a = analytical["Bx"]**2 + analytical["By"]**2
    analytical["Eb"] = _a
    _e = _p - _a
    error["Eb"] = _e

    # Magnetic field magnitude
    _p = np.sqrt(predicted["Eb"])
    predicted["B"] = _p
    _a = np.sqrt(analytical["Eb"])
    analytical["B"] = _a
    _e = _p - _a
    error["B"] = _e

    # Compute predicted, analytical, and error values for required derivatives
    # at each training point.
    txyv = tf.Variable(X_train)
    with tf.GradientTape(persistent=True) as tape1:
        Bxp = models[p.iBx](txyv)
        Byp = models[p.iBy](txyv)
    dBxp_dx = tape1.gradient(Bxp, txyv)[:, p.ix].numpy().reshape(nt, nx, ny)
    dByp_dy = tape1.gradient(Byp, txyv)[:, p.iy].numpy().reshape(nt, nx, ny)
    _p = dBxp_dx + dByp_dy
    predicted["divB"] = _p
    _a = np.zeros(_p.shape)
    analytical["divB"] = _a
    _e = _p - _a
    error["divB"] = _e

    # Compute RMS error values at each training time, and overall values.
    rms = {}
    RMS = {}
    iv = 0
    for (vname, _e) in error.items():
        _rms = np.sqrt(np.sum(_e**2)/_e.size)
        RMS[vname] = _rms
        rms[vname] = np.zeros(_e.shape[0])
        for it in range(_e.shape[0]):
            _rms = np.sqrt(np.sum(_e[it]**2)/_e[it].size)
            rms[vname][it] = _rms

    # Compute integrated magnetic energy over time.
    Ebtot = np.zeros(predicted["Eb"].shape[0])
    for it in range(predicted["Eb"].shape[0]):
        _Eb = predicted["Eb"][it]
        _Ebtot = np.sum(_Eb)
        Ebtot[it] = _Ebtot

    # ------------------------------------------------------------------------

    # Create the plots in a memory buffer.
    mpl.use("Agg")

    # Use LaTex in plots if requested.
    plt.rcParams.update({"text.usetex": args["usetex"]})

    # ------------------------------------------------------------------------

    # Plot the aggregate residual, data, and weighted loss histories.
    if verbose:
        print("Creating aggregate loss plot.")
    fig = create_loss_plot(L_res, L_dat, L)
    ax = fig.get_axes()[0]
    ax.set_title("Aggregate residual, data, and weighted loss")

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "L.png")
    fig.savefig(path)

    # ------------------------------------------------------------------------

    # Plot the per-model residual, data, and weighted loss histories.
    for iv in range(p.n_var):
        variable_name = p.dependent_variable_names[iv]
        variable_label = p.dependent_variable_labels[iv]
        if verbose:
            print(f"Creating loss plot for {variable_name}.")

        # Create the plot.
        fig = create_loss_plot(Lm_res[iv], Lm_dat[iv], Lm[iv])
        ax = fig.get_axes()[0]
        ax.set_title(f"{variable_label} residual, data, and weighted loss")

        # Save the plot to a PNG file.
        path = os.path.join(output_path, f"L_{variable_name}.png")
        fig.savefig(path)
        plt.close(fig)

    # ------------------------------------------------------------------------

    variable_names = p.dependent_variable_names + ["Eb", "B", "divB"]
    variable_labels = (
        p.dependent_variable_labels + ["$E_b$", "B", "divB"]
    )
    n_var = len(variable_names)

    # Create the predicted, analytical, and error movie for each variable.
    for iv in range(n_var):
        variable_name = variable_names[iv]
        variable_label = variable_labels[iv]
        if verbose:
            print(f"Creating PAE movie for {variable_name}.")

        # Create a directory for the PAE plots for this variable.
        pae_path = os.path.join(output_path, f"PAE_{variable_name}")
        os.mkdir(pae_path)

        # Plot for each training grid time.
        for it in range(nt):

            # Fetch the frame time and grid coordinates.
            t = T[it, 0, 0]
            _X = X[it].T
            _Y = Y[it].T

            # To get the proper orientation, reshape, transpose.
            P = predicted[variable_name][it].T
            A = analytical[variable_name][it].T
            E = error[variable_name][it].T

            # Create the plot.
            fig = create_PAE_plot(_X, _Y, P, A, E)

            # Tweak the frame title and error plot title.
            fig.suptitle(f"{variable_label}, t = {t:.2E} predicted, "
                         "analytical, and error "
                         f"(overall RMS={RMS[variable_name]:.2E})")
            fig.axes[2].set_title("Error (RMS = "
                                  f"{rms[variable_name][it]:.2E})")

            # Save the plot to a PNG file.
            path = os.path.join(pae_path, f"PAE_{variable_name}_{it:04d}.png")
            fig.savefig(path)
            plt.close(fig)

        # Assemble the frames into a movie.
        frame_pattern = os.path.join(pae_path, f"PAE_{variable_name}_%04d.png")
        movie_file = os.path.join(pae_path, f"PAE_{variable_name}.mp4")
        assemble_movie(frame_pattern, movie_file)

    # ------------------------------------------------------------------------

    # Make a PA movie of the magnetic field vectors.
    if verbose:
        print("Creating PA movie for magnetic field.")
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

    # Plot the RMS error as a function of time for each variable.
    t = T[:, 0, 0]
    for (iv, vname) in enumerate(variable_names):
        if verbose:
            print(f"Creating RMS error plot for {vname}.")

        # Create the plot.
        fig = create_rms_error_plot(t, rms[vname])

        # Tweak the title.
        fig.axes[0].set_title(f"{variable_names[iv]} RMS Error "
                              f"(overall = {RMS[vname]:.2E})")

        # Save the plot to a PNG file.
        path = os.path.join(output_path, f"RMS_{vname}.png")
        fig.savefig(path)
        plt.close(fig)

    # ------------------------------------------------------------------------

    # Plot the total magnetic energy as a function of time.
    if verbose:
        print("Creating total magnetic energy plot.")

    # Create the plot.
    fig = create_total_magnetic_energy_plot(t, Ebtot)

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "Ebtot.png")
    fig.savefig(path)
    plt.close(fig)

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
