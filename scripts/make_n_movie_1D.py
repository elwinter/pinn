#!/usr/bin/env python


"""Create a movie of the number density in a 1-D problem.

Create a movie of the number density in a 1-D problem.
The movie is created as an animated GIF. Individual frames are saved as
PNG files in the frames/ subdirectory.

The frames subdirectory, and the movie file, will be created in the current
directory.
"""


# Import standard modules.
import argparse
import os

# Import 3rd-party modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import project-specific modules.
from pinn import common
from pinn import standard_plots
from pinn import training_data


# Program constants and defaults

# Program description.
DESCRIPTION = (
    "Create a movie of the magnetic field magnitude as an animated heatmap."
)

# Default maximum and minimum values for B.
DEFAULT_NMIN = 0.0
DEFAULT_NMAX = 2.0

# Default epoch to use when selecting a trained model.
# -1 = use last epoch in results.
DEFAULT_EPOCH = -1

# Default points counts for movie frame generation.
DEFAULT_NT = 201
DEFAULT_NX = 201

# Default limits for t, x for movie frames.
DEFAULT_TMIN = 0.0
DEFAULT_TMAX = 1.6
DEFAULT_XMIN = 0.0
DEFAULT_XMAX = 2.0


def create_command_line_parser():
    """Create the command-line argument parser.

    Create the command-line argument parser.

    Parameters
    ----------
    None

    Returns
    -------
    parser : argparse.ArgumentParser
        Command-line argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--nmax", type=float, default=DEFAULT_NMAX,
        help="Maximum n value to plot (default: %(default)s)."
    )
    parser.add_argument(
        "--nmin", type=float, default=DEFAULT_NMIN,
        help="Minimum n value to plot (default: %(default)s)."
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--epoch", "-e", type=int, default=DEFAULT_EPOCH,
        help="Epoch for trained model to use for movie "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "--nt", type=int, default=DEFAULT_NT,
        help="Number of time points to use in movie (default: %(default)s)."
    )
    parser.add_argument(
        "--nx", type=int, default=DEFAULT_NX,
        help="Number of x points to use in movie (default: %(default)s)."
    )
    parser.add_argument(
        "--results_path", "-r", type=str, default=os.getcwd(),
        help="Directory containing model results (default: %(default)s)."
    )
    parser.add_argument(
        "--tmax", type=float, default=DEFAULT_TMAX,
        help="Maximum time value for movie (default: %(default)s)."
    )
    parser.add_argument(
        "--tmin", type=float, default=DEFAULT_TMIN,
        help="Minimum time value for movie (default: %(default)s)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "--xmax", type=float, default=DEFAULT_XMAX,
        help="Maximum x value for movie (default: %(default)s)."
    )
    parser.add_argument(
        "--xmin", type=float, default=DEFAULT_XMIN,
        help="Minimum x value for movie (default: %(default)s)."
    )
    return parser


def main():
    """Main program logic.

    The program starts here.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Set up the command-line parser.
    parser = create_command_line_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    nmax = args.nmax
    nmin = args.nmin
    debug = args.debug
    epoch = args.epoch
    nt = args.nt
    nx = args.nx
    results_path = args.results_path
    tmax = args.tmax
    tmin = args.tmin
    verbose = args.verbose
    xmax = args.xmax
    xmin = args.xmin
    if debug:
        print(f"args = {args}")
        print(f"nmax = {nmax}")
        print(f"nmin = {nmin}")
        print(f"debug = {debug}")
        print(f"epoch = {epoch}")
        print(f"nt = {nt}")
        print(f"nx = {nx}")
        print(f"results_path = {results_path}")
        print(f"tmax = {tmax}")
        print(f"tmin = {tmin}")
        print(f"verbose = {verbose}")
        print(f"xmax = {xmax}")
        print(f"xmin = {xmin}")

    # If -1 was specified for the model epoch, determine the last epoch.
    if epoch == -1:
        epoch = common.find_last_epoch(results_path)
    if debug:
        print(f"epoch = {epoch}")

    # Load the models for the specified epoch.
    base_path = os.path.join(results_path, "models", str(epoch))
    if debug:
        print(f"base_path = {base_path}")
    path = os.path.join(base_path, "model_n")
    if debug:
        print(f"path = {path}")
    model_n = tf.keras.models.load_model(path)
    if debug:
        print(f"model_n = {model_n}")

    # Create the evaluation grid for the movie frames.
    ng = [nt, nx]
    bg = [
        [tmin, tmax],
        [xmin, xmax],
    ]
    tx = training_data.create_training_points_gridded(ng, bg)

    # Create the frame directory.
    frame_dir = "./frames"
    os.mkdir(frame_dir)

    # Create the frames.
    mpl.use("Agg")
    for i in range(nt):
        i0 = i*nx
        i1 = i0 + nx
        if verbose:
            print(f"Creating frame {i} for time {tx[i0, 0]:0.2f}.")
        x = tx[i0:i1, 1]
        n = model_n(tx[i0:i1]).numpy()
        title = f"Density at t = {tx[i0, 0]:0.2f}"
        plt.clf()
        plt.xlim(xmin, xmax)
        plt.ylim(nmin, nmax)
        plt.title(title)
        plt.plot(x, n)
        frame_name = f"frame_{i:06d}.png"
        frame_path = os.path.join(frame_dir, frame_name)
        plt.savefig(frame_path)

    # Create the movie.
    os.system(f"convert -delay 10 -loop 0 {frame_dir}/frame_00*.png n.gif")


if __name__ == "__main__":
    """Begin main program."""
    main()
