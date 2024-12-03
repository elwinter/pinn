#!/usr/bin/env python


"""Create plots for GAMERA results for the loop2d problem.

AGU 2024 edition

Create plots for GAMERA results for the loop2d problem.

Note on notation: "PAE" -> predicted/analytical/error

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

# Import standard modules.
import copy
# from importlib import import_module
# import os
# import subprocess
import sys

# Import supplemental modules.
import h5py
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf

# Import project modules.
from pinn import common
sys.path.insert(0, "/homes/winteel1/research/src/kaipy-private/development/kaipy-private")
from kaipy import kaih5

# Program constants

# Program description
DESCRIPTION = (
    "Create plots for GAMERA results for loop2d problem."
)

# Default values for command-line arguments.
DEFAULT_ARGUMENTS = {
    "clobber": common.DEFAULT_ARGUMENTS["clobber"],
    "debug": common.DEFAULT_ARGUMENTS["debug"],
    "image_format": "png",
    "usetex": False,
    "verbose": common.DEFAULT_ARGUMENTS["verbose"],
    "results_path": None,
}

# # Name of problem
# PROBLEM_NAME = "loop2d_nPuxuyuzBxByBz"

# # Name of directory to hold output plots
# OUTPUT_DIR = "pinn1_plots"


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
        "--usetex", action="store_true",
        help="Use LaTeX in plots (default: %(default)s)"
    )
    parser.add_argument(
        "results_path",
        help="Path to directory containing results to plot."
    )
    return parser


# def make_PAE_plot(
#         Zp: np.ndarray, Za: np.ndarray, Ze: np.ndarray,
#         X: np.ndarray, Y: np.ndarray, **kwargs):
#     """Make a plot of the predicted and analytical solution, and error.

#     Make a plot of the predicted and analytical solution, and error.

#     All array arguments must have the same shape (ny, nx).

#     Parameters
#     ----------
#     Zp : np.ndarray, shape (ny, nx)
#         Predicted solution at each point.
#     Za : np.ndarray, shape (ny, nx)
#         Analytical solution at each point.
#     Ze : np.ndarray, shape (ny, nx)
#         Absolute error (Zp - Za) at each point.
#     X : np.ndarray, shape (ny, nx)
#         x-values for each training point.
#     Y : np.ndarray, shape (ny, nx)
#         y-values for each training point.
#     kwargs : dict
#         dict of additional keyword arguments

#     Returns
#     -------
#     fig : matplotlib.figure.Figure
#         Figure for plot.

#     Raises
#     ------
#     None
#     """
#     title = kwargs.get("title", "")
#     xlabel = kwargs.get("xlabel", "")
#     ylabel = kwargs.get("ylabel", "")
#     figsize = kwargs.get("figsize", None)
#     pvmin = kwargs.get("pvmin", None)
#     pvmax = kwargs.get("pvmax", None)
#     avmin = kwargs.get("avmin", None)
#     avmax = kwargs.get("avmax", None)
#     evmin = kwargs.get("evmin", None)
#     evmax = kwargs.get("evmax", None)

#     # Create the figure and Axes.
#     fig, axs = plt.subplots(1, 3, layout="constrained", figsize=figsize)
#     axp, axa, axe = axs

#     # Plot the predicted, analytical, and error solutions.
#     pcmp = axs[0].pcolormesh(X, Y, Zp, vmin=pvmin, vmax=pvmax)
#     pmean = np.mean(Zp)
#     axp.set_title(f"Predicted (mean = {pmean:.3E})")
#     axp.set_xlabel(xlabel)
#     axp.set_ylabel(ylabel)
#     fig.colorbar(pcmp, ax=axp, orientation="horizontal")

#     pcma = axs[1].pcolormesh(X, Y, Za, vmin=avmin, vmax=avmax)
#     amean = np.mean(Za)
#     axa.set_title(f"Analytical (mean = {amean:.3E})")
#     axa.set_xlabel(xlabel)
#     axa.set_ylabel(ylabel)
#     fig.colorbar(pcma, ax=axa, orientation="horizontal")

#     pcme = axs[2].pcolormesh(X, Y, Ze, vmin=evmin, vmax=evmax)
#     rms = np.sqrt(np.sum(Ze**2)/Ze.size)
#     axe.set_title(f"Error (RMS={rms:.3E})")
#     axe.set_xlabel(xlabel)
#     axe.set_ylabel(ylabel)
#     fig.colorbar(pcme, ax=axe, orientation="horizontal")

#     # Set common plot options.
#     for ax in axs:
#         ax.grid()
#         ax.set_aspect("equal")

#     # Decorate the plot.
#     fig.suptitle(title)

#     # Return the figure.
#     return fig


# def make_PAE_B_plot(
#         Bxp: np.ndarray, Byp: np.ndarray,
#         Bxa: np.ndarray, Bya: np.ndarray,
#         Bxe: np.ndarray, Bye: np.ndarray,
#         X: np.ndarray, Y: np.ndarray, **kwargs):
#     """Make a plot of the predicted and analytical B field, and error.

#     Make a plot of the predicted and analytical magnetic field, and error.

#     All array arguments must have the same shape (ny, nx).

#     Parameters
#     ----------
#     Bxp : np.ndarray, shape (ny, nx)
#         Predicted value of Bx.
#     Byp : np.ndarray, shape (ny, nx)
#         Predicted value of By.
#     Bxa : np.ndarray, shape (ny, nx)
#         Analytical value of Bx.
#     Bya : np.ndarray, shape (ny, nx)
#         Analytical value of By.
#     Bxe : np.ndarray, shape (ny, nx)
#         Absolute error (Bxp - Bxa) at each point.
#     Bye : np.ndarray, shape (ny, nx)
#         Absolute error (Byp - Bya) at each point.
#     X : np.ndarray, shape (ny, nx)
#         x-values for each training point.
#     Y : np.ndarray, shape (ny, nx)
#         y-values for each training point.
#     kwargs : dict
#         dict of additional keyword arguments

#     Returns
#     -------
#     fig : matplotlib.figure.Figure
#         Figure for plot.

#     Raises
#     ------
#     None
#     """
#     title = kwargs.get("title", "")
#     xlabel = kwargs.get("xlabel", "")
#     ylabel = kwargs.get("ylabel", "")
#     figsize = kwargs.get("figsize", None)

#     # Create the figure and Axes.
#     fig, axs = plt.subplots(1, 3, layout="constrained", figsize=figsize)
#     axp, axa, axe = axs

#     # Plot the fields.
#     axp.quiver(X, Y, Bxp, Byp)
#     axp.set_title("Predicted")
#     axa.quiver(X, Y, Bxa, Bya)
#     axa.set_title("Analytical")
#     axe.quiver(X, Y, Bxe, Bye)
#     axe.set_title("Error")

#     # Set common plot options.
#     for ax in axs:
#         ax.grid()
#         ax.set_aspect("equal")
#         ax.set_xlabel("$x$")
#         ax.set_ylabel("$y$")

#     # Decorate the plot.
#     fig.suptitle(title)

#     # Return the figure.
#     return fig


# def assemble_movie(movie_file: str, frame_pattern: str, frame_rate: int):
#     """Assemble a movie from individual frames.

#     Assemble a movie from individual frames.

#     Parameters
#     ----------
#     movie_file : str
#         Path to movie file.
#     frame_pattern : str
#         glob pattern for frame file names.
#     frame_rate : int
#         Frame rate (frames/second).

#     Returns
#     -------
#     None

#     Raises
#     ------
#     None
#     """
#     cmd = (
#         f"ffmpeg -r {frame_rate} -i {frame_pattern} {movie_file}"
#     )
#     subprocess.run(cmd, shell=True, check=True)


def gamera_plots(args: dict):
    """Primary entry point for gamera plots.

    This is the main program code for making gamera plots. This function can be
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
    usetex = args["usetex"]
    verbose = args["verbose"]
    results_path = args["results_path"]

    # ------------------------------------------------------------------------

    # Create the output directory.
    if verbose:
        print("Creating output directory.")
    output_dir = common.create_directory("gamera_plots", clobber)
    if debug:
        print(f"output_dir = {output_dir}")

    # ------------------------------------------------------------------------

    # Load the time steps.
    if verbose:
        print(f"Loading time steps from {results_path}")
    n_steps, step_ids = kaih5.cntSteps(results_path)
    if debug:
        print(f"n_steps = {n_steps}")
        print(f"step_ids = {step_ids}")

    # Load the locations of the GAMERA grid points. These are independent
    # variables only.
    if verbose:
        print(f"Reading coordinate grid from {results_path}.")
    with h5py.File(results_path, "r") as f:
        Xg = f["X"][...]
        Yg = f["Y"][...]
    if debug:
        print(f"Xg = {Xg}")
        print(f"Yg = {Yg}")

    # ------------------------------------------------------------------------

#     # Compute details of the training grid.

#     # Extract grid counts.
#     nt = column_descriptions_train["t"]["n"]
#     nx = column_descriptions_train["x"]["n"]
#     ny = column_descriptions_train["y"]["n"]
#     nxy = nx*ny
#     if debug:
#         print(f"nt = {nt}")
#         print(f"nx = {nx}")
#         print(f"ny = {ny}")
#         print(f"nxy = {nxy}")

#     # Compute grid cell size.
#     xmin = column_descriptions_train["x"]["min"]
#     xmax = column_descriptions_train["x"]["max"]
#     dx = (xmax - xmin)/(nx - 1)
#     ymin = column_descriptions_train["y"]["min"]
#     ymax = column_descriptions_train["y"]["max"]
#     dy = (ymax - ymin)/(ny - 1)
#     dxy = dx*dy
#     if debug:
#         print(f"dx = {dx}")
#         print(f"dy = {dy}")
#         print(f"dxy = {dxy}")

#     # Extract and reshape the x- and y-grid values from the first time step.
#     # The rest of this code assumes the grid to be the same for all time steps.
#     # NOTE: Extracted values of data must also use this reshape(ny, nx).T.
#     # This reshaping is required by the order in which the points are stored
#     # in the training grid file.
#     Xg = X_train[:nxy, p.ix].reshape(ny, nx).T
#     Yg = X_train[:nxy, p.iy].reshape(ny, nx).T
#     if debug:
#         print(f"Xg = {Xg}")
#         print(f"Yg = {Yg}")

#     # ------------------------------------------------------------------------

#     # Naming conventions:
#     # terminal "s" -> list of np.ndarray or tf.Variable/Tensor
#     # terminal "_tf" -> A TensorFlow Variable or Tensor

    # ------------------------------------------------------------------------

#     # Compute the predicted and analytical solutions and derivatives, and
#     # errors at the training points.
#     if verbose:
#         print("Computing predicted and analytical solutions and derivatives, "
#               "and error for training points.")
#     # Xt_tf is tf.Variable, shape (n_train, p.n_dim).
#     Xt_tf = tf.Variable(X_train)
#     with tf.GradientTape(persistent=True) as tape1:
#         # Ytps is list of p.n_var tf.Tensor, each shape (n_train, 1).
#         Ytps = [model(Xt_tf) for model in models]
#     # dYtp_dXts is list of p.n_var tf.Tensor, each shape (n_train, p.n_dim).
#     dYtp_dXts = [tape1.gradient(Y, Xt_tf) for Y in Ytps]
#     if debug:
#         print(f"Ytps = {Ytps}")
#         print(f"dYtp_dXts = {dYtp_dXts}")

#     # Gather predicted values for each model into np.ndarray.
#     # Convert eachh Tensor to np.ndarray.
#     # Ytps is now list of p.n_var np.ndaray of float, each shape (n_train, 1).
#     Ytps = [Y.numpy() for Y in Ytps]
#     # Ytps is now np.ndaray of float, each shape (n_train, p.n_var).
#     Ytp = np.hstack(Ytps)
#     # dYtp_dXts is now list of p.n_var np.ndarray of float, each shape
#     # (n_train, p.n_dim).
#     dYtp_dXts = [dY_dX.numpy() for dY_dX in dYtp_dXts]
#     if debug:
#         print(f"Ytps = {Ytps}")
#         print(f"Ytp = {Ytp}")
#         print(f"dYtp_dXts = {dYtp_dXts}")

#     # Extract individual columns from training grid.
#     # Each is np.ndarray, shape (n_train, 1)
#     t = X_train[:, 0]
#     x = X_train[:, 1]
#     y = X_train[:, 2]

#     # Compute analytical values and error of the dependent variables at the
#     # training points.
#     # Yta is list of p.n_var np.ndarray of float, shape (n_train, 1).
#     Yta = [f(t, x, y).reshape(n_train, 1) for f in p.Y_analytical]
#     # Yta is now np.ndarray of float, shape (n_train, p.n_var).
#     # Yte is np.ndarray of float, shape (n_train, p.n_var).
#     Yta = np.hstack(Yta)
#     Yte = Ytp - Yta
#     if debug:
#         print(f"Ytp = {Ytp}")
#         print(f"Yta = {Yta}")
#         print(f"Yte = {Yte}")

#     # Compute analytical values of the derivatives of the dependent variables
#     # wrt the independent variables at the training points.
#     # Each is np.ndarray, shape (n_train, 1).
#     dBx_dxa = p.dBx_dx_analytical(t, x, y)
#     dBx_dya = p.dBx_dy_analytical(t, x, y)
#     dBy_dya = p.dBy_dy_analytical(t, x, y)
#     dBy_dxa = p.dBy_dx_analytical(t, x, y)
#     if debug:
#         print(f"dBx_dxa = {dBx_dxa}")
#         print(f"dBx_dya = {dBx_dya}")
#         print(f"dBy_dya = {dBy_dya}")
#         print(f"dBy_dxa = {dBy_dxa}")

#     # Compute the predicted and analytical derivatives and magnetic divergence,
#     # and error, at the training points.
#     if verbose:
#         print("Computing predicted and analytical magnetic divergence, and "
#               "error at training points.")
#     # Each is np.ndarray of float, shape (n_train,).
#     dBx_dxp = dYtp_dXts[p.iBx][:, p.ix]
#     dBx_dxe = dBx_dxp - dBx_dxa
#     dBy_dyp = dYtp_dXts[p.iBy][:, p.iy]
#     dBy_dye = dBy_dyp - dBy_dya
#     divBp = dBx_dxp**2 + dBy_dyp**2
#     divBa = np.zeros_like(divBp)
#     divBe = divBp - divBa
#     if debug:
#         print(f"dBx_dxp = {dBx_dxp}")
#         print(f"dBx_dxe = {dBx_dxe}")
#         print(f"dBy_dyp = {dBy_dyp}")
#         print(f"dBy_dye = {dBy_dye}")
#         print(f"divBp = {divBp}")
#         print(f"divBa = {divBa}")
#         print(f"divBe = {divBe}")

#     # Compute the predicted and analytical magnetic pressure and error, at the
#     # training points.
#     if verbose:
#         print("Computing predicted and analytical magnetic pressure, and "
#               "error at training points.")
#     # Each is np.ndarray of float, shape (n_train,).
#     PBp = Ytp[:, p.iBx]**2 + Ytp[:, p.iBy]**2
#     PBa = Yta[:, p.iBx]**2 + Yta[:, p.iBy]**2
#     PBe = PBp - PBa
#     if debug:
#         print(f"PBp = {PBp}")
#         print(f"PBa = {PBa}")
#         print(f"PBe = {PBe}")

#     # Compute the predicted and analytical current density in the z-direction.
#     # The analytical current density is 0 everywhere except along the z-axis.
#     if verbose:
#         print("Computing predicted and analytical z-current density, and "
#               "error at training points.")
#     # Each is np.ndarray of float, shape (n_train,).
#     uxp = Ytp[:, p.iux]
#     uyp = Ytp[:, p.iuy]
#     Bxp = Ytp[:, p.iBx]
#     Byp = Ytp[:, p.iBy]
#     Jzp = uxp*Byp - uyp*Bxp
#     Jza = np.zeros_like(Jzp)
#     Jze = Jzp - Jza
#     if debug:
#         print(f"uxp = {uxp}")
#         print(f"uyp = {uyp}")
#         print(f"Bxp = {Bxp}")
#         print(f"Byp = {Byp}")
#         print(f"Jzp = {Jzp}")
#         print(f"Jza = {Jza}")
#         print(f"Jze = {Jze}")

    # # ------------------------------------------------------------------------

    # # Create the plots in a memory buffer.
    # mpl.use("Agg")

    # # ------------------------------------------------------------------------

    # # Set shared parameters for movies and frames.

    # # Compute the number of frames in each movie as the number of time values.
    # # This code assumes the same xy points used at each time.
    # n_frames = nt

    # # Specify the frame rate in frames/second.
    # frame_rate = 10

    # # Constant plot parameters.
    # figsize = (12, 5)
    # xlabel = p.independent_variable_labels[p.ix]
    # ylabel = p.independent_variable_labels[p.iy]

    # # Use LaTex in plots (not available on mollie).
    # plt.rcParams.update({
    #     "text.usetex": usetex,
    #     "font.family": "Times New Roman",
    #     "font.size": 14,
    # })

#     # Specify value limits for pcolormesh() for each model.
#     vmin = {
#         'n': 0.99,
#         'P': 0.98,
#         'ux': 0.86,
#         'uy': 0.499,
#         'uz': -1e-3,
#         'Bx': -5e-3,
#         'By': -5e-3,
#         'Bz': -1e-3,
#         'divB': -1e-3,
#         'PB': 9e-7,
#         'Jz': -1e-3,
#     }
#     vmax = {
#         'n': 1.01,
#         'P': 1.02,
#         'ux': 0.87,
#         'uy': 0.501,
#         'uz': 1e-3,
#         'Bx': 5e-3,
#         'By': 5e-3,
#         'Bz': 1e-3,
#         'divB': 1e-3,
#         'PB': 1.1e-6,
#         'Jz': 1e-3,
#     }
#     evmin = {
#         'n': -5e-3,
#         'P': -1e-1,
#         'ux': -1e-2,
#         'uy': -1e-3,
#         'uz': -1e-6,
#         'Bx': -1e-4,
#         'By': -1e-4,
#         'Bz': -1e-4,
#         'divB': -1e-3,
#         'PB': -1e-7,
#         'Jz': -1e-3,
#     }
#     evmax = {
#         'n': 5e-3,
#         'P': 1e-1,
#         'ux': 1e-2,
#         'uy': 1e-3,
#         'uz': 1e-6,
#         'Bx': 1e-4,
#         'By': 1e-4,
#         'Bz': 1e-4,
#         'divB': 1e-3,
#         'PB': 1e-7,
#         'Jz': 1e-3,
#     }

#     # ------------------------------------------------------------------------

#     # Make a movie of the predicted and analytical solutions, and the error,
#     # (PAE plots) for each model.

#     # Make a plot for each model.
#     for iv in range(p.n_var):

#         # Extract the variable name and label.
#         variable_name = p.dependent_variable_names[iv]
#         variable_label = p.dependent_variable_labels[iv]

#         if verbose:
#             print("Creating predicted/analytical/error (PAE) movie for "
#                   f"{variable_name}.")

#         # Create the directory for the frames for this variable.
#         frame_dir = os.path.join(output_dir, f"frames_{variable_name}")
#         os.mkdir(frame_dir)

#         # Create the frames.
#         if verbose:
#             print(f"Creating frames for {variable_name} movie.")
#         for i_frame in range(n_frames):
#             if verbose:
#                 print(f"Creating {variable_name} frame {i_frame}.")

#             # Extract the time of the frame.
#             t = X_train[i_frame*nxy, 0]

#             # Extract the data for the frame.
#             i1 = i_frame*nxy
#             i2 = (i_frame + 1)*nxy
#             # NOTE: Needs same reshape().T used by grid points.
#             # X = X_train[i1:i2, p.ix].reshape(ny, nx).T
#             # Y = X_train[i1:i2, p.iy].reshape(ny, nx).T
#             Zp = Ytp[i1:i2, iv].reshape(ny, nx).T
#             Za = Yta[i1:i2, iv].reshape(ny, nx).T
#             Ze = Yte[i1:i2, iv].reshape(ny, nx).T

#             # Compute the frame title.
#             title = f"{variable_label} at t = {t:0.3E}"
#             fig = make_PAE_plot(
#                 Zp, Za, Ze, Xg, Yg,
#                 title=title, xlabel=xlabel, ylabel=ylabel, figsize=figsize,
#                 pvmin=vmin[variable_name], pvmax=vmax[variable_name],
#                 avmin=vmin[variable_name], avmax=vmax[variable_name],
#                 evmin=evmin[variable_name], evmax=evmax[variable_name]
#             )

#             # Save the plot to a file.
#             path = os.path.join(
#                 frame_dir, f"{variable_name}-{i_frame:06d}.{image_format}")
#             if verbose:
#                 print(f"Saving {path}.")
#             plt.savefig(path)

#             # Close the figure.
#             plt.close(fig)

#             # End of frame loop.

#         # Assemble the frames into a movie.
#         if verbose:
#             print(f"Assembling movie for {variable_name}.")
#         frame_pattern = os.path.join(
#             frame_dir, f"{variable_name}-%06d.{image_format}"
#         )
#         movie_file = os.path.join(output_dir, f"{variable_name}.mp4")
#         assemble_movie(movie_file, frame_pattern, frame_rate)

#         # End of variable loop.

#     # ------------------------------------------------------------------------

#     # Make a movie of the predicted and analytical magnetic field vectors, and
#     # the error.
#     if verbose:
#         print("Creating predicted/analytical/error movie for xy-magnetic "
#               "field.")

#     # Create the directory for the frames.
#     variable_name = "BxBy"
#     frame_dir = os.path.join(output_dir, f"frames_{variable_name}")
#     os.mkdir(frame_dir)

#     # Create the frames.
#     if verbose:
#         print(f"Creating frames for {variable_name} movie.")
#     for i_frame in range(n_frames):
#         if verbose:
#             print(f"Creating BxBy frame {i_frame}.")

#         # Extract the time of the frame.
#         t = X_train[i_frame*nxy, p.it]

#         # Extract the data for the frame.
#         i1 = i_frame*nxy
#         i2 = (i_frame + 1)*nxy
#         # NOTE: Needs same reshape().T used by grid points,.
#         Bxp = Ytp[i1:i2, p.iBx].reshape(ny, nx).T
#         Byp = Ytp[i1:i2, p.iBy].reshape(ny, nx).T
#         Bxa = Yta[i1:i2, p.iBx].reshape(ny, nx).T
#         Bya = Yta[i1:i2, p.iBy].reshape(ny, nx).T
#         Bxe = Bxp - Bxa
#         Bye = Byp - Bya

#         # Compute the frame title.
#         title = f"Magnetic field at t = {t:0.3E}"

#         # Plot the magnetic field vectors.
#         fig = make_PAE_B_plot(Bxp, Byp, Bxa, Bya, Bxe, Bye, Xg, Yg,
#                               title=title, figsize=figsize)

#         # Save the plot to a file.
#         path = os.path.join(
#             frame_dir, f"BxBy-{i_frame:06d}.{image_format}")
#         if verbose:
#             print(f"Saving {path}.")
#         plt.savefig(path)

#         # Close the figure.
#         plt.close(fig)

#         # End of frame loop.

#     # Assemble the frames into a movie.
#     if verbose:
#         print(f"Assembling movie for {variable_name}.")
#     frame_pattern = os.path.join(
#         frame_dir, f"{variable_name}-%06d.{image_format}"
#     )
#     movie_file = os.path.join(output_dir, f"{variable_name}.mp4")
#     assemble_movie(movie_file, frame_pattern, frame_rate)

#     # ------------------------------------------------------------------------

#     # Make a movie of the predicted and analytical magnetic divergence, and
#     # the error.
#     if verbose:
#         print("Creating predicted/analytical/error movie for magnetic "
#               "divergence.")

#     # Create the directory for the frames.
#     variable_name = "divB"
#     frame_dir = os.path.join(output_dir, f"frames_{variable_name}")
#     os.mkdir(frame_dir)

#     # Create the frames.
#     if verbose:
#         print(f"Creating frames for {variable_name} movie.")
#     for i_frame in range(n_frames):
#         if verbose:
#             print(f"Creating divB frame {i_frame}.")

#         # Extract the time of the frame.
#         t = X_train[i_frame*nxy, p.it]

#         # Extract the data for the frame.
#         i1 = i_frame*nxy
#         i2 = (i_frame + 1)*nxy
#         # NOTE: Needs same reshape().T used by grid points,.
#         Zp = divBp[i1:i2].reshape(ny, nx).T
#         Za = divBa[i1:i2].reshape(ny, nx).T
#         Ze = divBe[i1:i2].reshape(ny, nx).T

#         # Compute the frame title.
#         if usetex:
#             variable_label = r"$\nabla \cdot \mathbf B$"
#         else:
#             variable_label = "divB"
#         title = f"{variable_label} at t = {t:0.3E}"

#         # Plot the magnetic field vectors.
#         fig = make_PAE_plot(
#             Zp, Za, Ze, Xg, Yg, title=title, figsize=figsize,
#             pvmin=vmin[variable_name], pvmax=vmax[variable_name],
#             avmin=vmin[variable_name], avmax=vmax[variable_name],
#             evmin=evmin[variable_name], evmax=evmax[variable_name]
#         )

#         # Save the plot to a file.
#         path = os.path.join(
#             frame_dir, f"{variable_name}-{i_frame:06d}.{image_format}")
#         if verbose:
#             print(f"Saving {path}.")
#         plt.savefig(path)

#         # Close the figure.
#         plt.close(fig)

#         # End of frame loop.

#     # Assemble the frames into a movie.
#     if verbose:
#         print(f"Assembling movie for {variable_name}.")
#     frame_pattern = os.path.join(
#         frame_dir, f"{variable_name}-%06d.{image_format}"
#     )
#     movie_file = os.path.join(output_dir, f"{variable_name}.mp4")
#     assemble_movie(movie_file, frame_pattern, frame_rate)

#     # ------------------------------------------------------------------------

#     # Make a movie of the predicted and analytical magnetic pressure, and
#     # the error.
#     if verbose:
#         print("Creating predicted/analytical/error movie for magnetic "
#               "pressure.")

#     # Create the directory for the frames.
#     variable_name = "PB"
#     frame_dir = os.path.join(output_dir, f"frames_{variable_name}")
#     os.mkdir(frame_dir)

#     # Create the frames.
#     if verbose:
#         print(f"Creating frames for {variable_name} movie.")
#     for i_frame in range(n_frames):
#         if verbose:
#             print(f"Creating PB frame {i_frame}.")

#         # Extract the time of the frame.
#         t = X_train[i_frame*nxy, p.it]

#         # Extract the data for the frame.
#         i1 = i_frame*nxy
#         i2 = (i_frame + 1)*nxy
#         # NOTE: Needs same reshape().T used by grid points,.
#         Zp = PBp[i1:i2].reshape(ny, nx).T
#         Za = PBa[i1:i2].reshape(ny, nx).T
#         Ze = PBe[i1:i2].reshape(ny, nx).T

#         # Compute the frame title.
#         if usetex:
#             variable_label = r"$P_B$"
#         else:
#             variable_label = "Magnetic pressure"
#         title = f"{variable_label} at t = {t:0.3E}"

#         # Plot the magnetic field vectors.
#         fig = make_PAE_plot(
#             Zp, Za, Ze, Xg, Yg, title=title, figsize=figsize,
#             pvmin=vmin[variable_name], pvmax=vmax[variable_name],
#             avmin=vmin[variable_name], avmax=vmax[variable_name],
#             evmin=evmin[variable_name], evmax=evmax[variable_name]
#         )

#         # Save the plot to a file.
#         path = os.path.join(
#             frame_dir, f"{variable_name}-{i_frame:06d}.{image_format}")
#         if verbose:
#             print(f"Saving {path}.")
#         plt.savefig(path)

#         # Close the figure.
#         plt.close(fig)

#         # End of frame loop.

#     # Assemble the frames into a movie.
#     if verbose:
#         print(f"Assembling movie for {variable_name}.")
#     frame_pattern = os.path.join(
#         frame_dir, f"{variable_name}-%06d.{image_format}"
#     )
#     movie_file = os.path.join(output_dir, f"{variable_name}.mp4")
#     movie_file = os.path.join(output_dir, f"{variable_name}.mp4")
#     assemble_movie(movie_file, frame_pattern, frame_rate)

#     # ------------------------------------------------------------------------

#     # Make a movie of the predicted and analytical z-current density, and
#     # the error.
#     if verbose:
#         print("Creating predicted/analytical/error movie for z-current "
#               "density.")

#     # Create the directory for the frames.
#     variable_name = "Jz"
#     frame_dir = os.path.join(output_dir, f"frames_{variable_name}")
#     os.mkdir(frame_dir)

#     # Create the frames.
#     if verbose:
#         print(f"Creating frames for {variable_name} movie.")
#     for i_frame in range(n_frames):
#         if verbose:
#             print(f"Creating Jz frame {i_frame}.")

#         # Extract the time of the frame.
#         t = X_train[i_frame*nxy, p.it]

#         # Extract the data for the frame.
#         i1 = i_frame*nxy
#         i2 = (i_frame + 1)*nxy
#         # NOTE: Needs same reshape().T used by grid points,.
#         Zp = Jzp[i1:i2].reshape(ny, nx).T
#         Za = Jza[i1:i2].reshape(ny, nx).T
#         Ze = Jze[i1:i2].reshape(ny, nx).T

#         # Compute the frame title.
#         if usetex:
#             variable_label = r"$J_z$"
#         else:
#             variable_label = "Jz"
#         title = f"{variable_label} at t = {t:0.3E}"

#         # Plot the z-current density.
#         fig = make_PAE_plot(
#             Zp, Za, Ze, Xg, Yg, title=title, figsize=figsize,
#             pvmin=vmin[variable_name], pvmax=vmax[variable_name],
#             avmin=vmin[variable_name], avmax=vmax[variable_name],
#             evmin=evmin[variable_name], evmax=evmax[variable_name]
#         )

#         # Save the plot to a file.
#         path = os.path.join(
#             frame_dir, f"{variable_name}-{i_frame:06d}.{image_format}")
#         if verbose:
#             print(f"Saving {path}.")
#         plt.savefig(path)

#         # Close the figure.
#         plt.close(fig)

#         # End of frame loop.

#     # Assemble the frames into a movie.
#     if verbose:
#         print(f"Assembling movie for {variable_name}.")
#     frame_pattern = os.path.join(
#         frame_dir, f"{variable_name}-%06d.{image_format}"
#     )
#     movie_file = os.path.join(output_dir, f"{variable_name}.mp4")
#     assemble_movie(movie_file, frame_pattern, frame_rate)

    # Return normally.
    return 0


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
    return_code = gamera_plots(args)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
