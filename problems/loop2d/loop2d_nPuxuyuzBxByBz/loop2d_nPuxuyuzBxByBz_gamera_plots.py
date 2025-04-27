#!/usr/bin/env python

"""Create plots for gamera results for loop2d_nPuxuyuzBxByBz problem.

Create plots for gamera results for loop2d_nPuxuyuzBxByBz problem.

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
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import project modules.
from kaipy import kaiH5
from kaipy import kaiTools
import pinn.common


# Program constants

# Program description
DESCRIPTION = (
    "Create plots for gamera results for loop2d_nPuxuyuzBxByBz problem."
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
OUTPUT_DIR = "gamera_plots"

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
        "pinn1_results_path",
        help="Path to directory containing pinn1 results."
    )
    parser.add_argument(
        "gamera_results_path",
        help="Path to file containing gamera results."
    )
    return parser


def read_gamera_grid(path: str) -> np.ndarray:
    """Read time and coordinate values from a gamera results file.

    Read time and coordinate values from a gamera results file.

    Note that the values are combined and rearranged so that they are returned
    in the same format as the pinn1 results read with np.loadtxt().

    Parameters
    ----------
    path : str
        Path to gamera results file.

    Returns
    -------
    TXY : np.ndarray, shape (nt*nx*ny, 3)
        (t, x, y) values for each gamera time and grid point.
    nt, nx, ny : int
        Count of points in t, x, y dimensions.

    Raises
    ------
    None
    """
    # Load the time steps.
    t = kaiH5.getTs(path, "time")
    nt = t.shape[0]

    # Load the locations of the GAMERA grid points. These are independent
    # variables only.
    with h5py.File(path, "r") as f:
        Xg = f["X"][...]
        Yg = f["Y"][...]

    # Compute the coordinates of the grid cell centers.
    Xc = kaiTools.to_center2D(Xg)
    Yc = kaiTools.to_center2D(Yg)

    # Are these right?
    nx = Xc.shape[1]
    ny = Yc.shape[0]

    # Stack the arrays to form the same structure as pinn1 training data.
    T = np.repeat(t, nx*ny)
    X = np.tile(Xc.T.flatten(), nt)
    Y = np.tile(Yc.T.flatten(), nt)
    TXY = np.vstack([T, X, Y]).T

    # Return the reshaped data.
    return TXY, nt, nx, ny


def load_gamera_variable(path: str, variable_name : str) -> np.ndarray:
    """Read all values for a variable from a gamera results file.

    Read all values for a variable from a gamera results file.

    Note that the values are combined and rearranged so that they are returned
    in the same format as the pinn1 results read with np.loadtxt().

    Parameters
    ----------
    path : str
        Path to gamera results file.
    variable_name : str
        Name of variable to load.

    Returns
    -------
    gamera_values : nd.array of shape (nt, nx, ny)
        Predicted variable values for each gamera time and grid point for the
        requested variable.

    Raises
    ------
    None
    """
    # Fetch the step IDs.
    n_steps, step_ids = kaiH5.cntSteps(path)
    gamera_values = []
    for step in step_ids:
        # NOTE: PullVar() does a transpose before returning data.
        v = kaiH5.PullVar(path, variable_name, step)
        gamera_values.append(v)
    gamera_values = np.stack(gamera_values, axis=0)

    # Return the variable.
    return gamera_values


def load_gamera_predicted(path: str) -> dict:
    """Read predicted values from a gamera results file.

    Read predicted values from a gamera results file.

    Note that the values are combined and rearranged so that they are returned
    in the same format as the pinn1 results read with np.loadtxt().

    Parameters
    ----------
    path : str
        Path to gamera results file.

    Returns
    -------
    predicted : dict of nd.array of shape (nt, nx, ny)
        Predicted variable values for each gamera time and grid point, key is
        variable name. Gamera names are mapped to pinn names for the keys.

    Raises
    ------
    None
    """
    # Initialize the dictionary.
    predicted = {}

    # Load the variables.
    predicted["n"] = load_gamera_variable(path, "D")
    predicted["P"] = load_gamera_variable(path, "P")
    predicted["ux"] = load_gamera_variable(path, "Vx")
    predicted["uy"] = load_gamera_variable(path, "Vy")
    predicted["uz"] = load_gamera_variable(path, "Vz")
    predicted["Bx"] = load_gamera_variable(path, "Bx")
    predicted["By"] = load_gamera_variable(path, "By")
    predicted["Bz"] = load_gamera_variable(path, "Bz")

    # Return the predicted values.
    return predicted


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


def gamera_plots(**kwargs) -> int:
    """Create gamera plots for the loop2d_nPuxuyuzBxByBz problem.

    Create gamera plots for the loop2d_nPuxuyuzBxByBz problem.

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
    pinn1_results_path = args["pinn1_results_path"]
    gamera_results_path = args["gamera_results_path"]
    if debug:
        print(f"debug = {debug}")
        print(f"verbose = {verbose}")
        print(f"pinn1_results_path = {pinn1_results_path}")
        print(f"gamera_results_path = {gamera_results_path}")

    # ------------------------------------------------------------------------

    # Add the pinn1 results directory to the module search path.
    sys.path.append(pinn1_results_path)

    # Import the problem definition from the pinn1 results directory.
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

    # Load the gamera grid.
    TXY_gamera, nt, nx, ny = read_gamera_grid(gamera_results_path)
    if debug:
        print(f"TXY_gamera = {TXY_gamera}")
        print(f"(nt, nx, ny) = ({nt}, {nx}, {ny})")

    # ------------------------------------------------------------------------

    # Compute derived values.

    # Extract the T, X, and Y values for the gamera points.
    T = TXY_gamera[:, p.it].reshape(nt, nx, ny)
    X = TXY_gamera[:, p.ix].reshape(nt, nx, ny)
    Y = TXY_gamera[:, p.iy].reshape(nt, nx, ny)
    if debug:
        print(f"T = {T}")
        print(f"X = {X}")
        print(f"Y = {Y}")

    # Load predicted variables at each time step.
    predicted = load_gamera_predicted(gamera_results_path)

    # Compute analytical, and error values for each variable at each grid
    # point. All are shape (nt, nx, ny).
    analytical = {}
    error = {}
    for iv in range(p.n_var):
        variable_name = p.dependent_variable_names[iv]
        _a = p.analytical_solutions[iv](T, X, Y)
        analytical[variable_name] = _a
        _e = predicted[variable_name] - _a
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

    # Compute predicted values for required derivatives at each gamera point.
    # Assume uniform spacing in each dimension.
    dx = X[0, 1, 0] - X[0, 0, 0]
    dy = Y[0, 0, 1] - Y[0, 0, 0]
    dBxp_dx = []
    dByp_dy = []
    for it in range(nt):
        _dBx_dx = np.gradient(predicted["Bx"][it], dy, dx)[1]
        dBxp_dx.append(_dBx_dx)
        _dBy_dy = np.gradient(predicted["By"][it], dy, dx)[0]
        dByp_dy.append(_dBy_dy)
    dBxp_dx = np.stack(dBxp_dx, axis=0)
    dByp_dy = np.stack(dByp_dy, axis=0)

    # Compute the predicted, analytical, and error magnetic divergence.
    _p = dBxp_dx + dByp_dy
    predicted["divB"] = _p
    _a = np.zeros(_p.shape)
    analytical["divB"] = _a
    _e = _p - _a
    error["divB"] = _e

    if debug:
        print(f"predicted = {predicted}")
        print(f"analytical = {analytical}")
        print(f"error = {error}")

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

    # Use LaTeX in plots if requested.
    plt.rcParams.update({"text.usetex": args["usetex"]})

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
    return_code = gamera_plots(**args)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
