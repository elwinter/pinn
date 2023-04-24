#!/usr/bin/env python


"""Create a movie of the magnetic field as an animated quiver plot.

Create a movie of the magnetic field as an animated quiver plot.
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
    "Create a movie of the magnetic field as an animated quiver plot."
)

# Default epoch to use when selecting a trained model.
# -1 = use last epoch in results.
DEFAULT_EPOCH = -1

# Default points counts for movie frame generation.
DEFAULT_NT = 51
DEFAULT_NX = 51
DEFAULT_NY = 51

# Default limits for t, x, y for movie frames.
DEFAULT_TMIN = 0.0
DEFAULT_TMAX = 1.0
DEFAULT_XMIN = -1.0
DEFAULT_XMAX = 1.0
DEFAULT_YMIN = -1.0
DEFAULT_YMAX = 1.0

# Tick counts for x- and y-axes in quiver plot.
QUIVER_N_X_TICKS = 5
QUIVER_N_Y_TICKS = 5


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
        "--ny", type=int, default=DEFAULT_NY,
        help="Number of y points to use in movie (default: %(default)s)."
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
    parser.add_argument(
        "--ymax", type=float, default=DEFAULT_YMAX,
        help="Maximum y value for movie (default: %(default)s)."
    )
    parser.add_argument(
        "--ymin", type=float, default=DEFAULT_YMIN,
        help="Minimum y value for movie (default: %(default)s)."
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
    debug = args.debug
    epoch = args.epoch
    nt = args.nt
    nx = args.nx
    ny = args.ny
    results_path = args.results_path
    tmax = args.tmax
    tmin = args.tmin
    verbose = args.verbose
    xmax = args.xmax
    xmin = args.xmin
    ymax = args.ymax
    ymin = args.ymin
    if debug:
        print(f"args = {args}")
        print(f"debug = {debug}")
        print(f"epoch = {epoch}")
        print(f"nt = {nt}")
        print(f"nx = {nx}")
        print(f"ny = {ny}")
        print(f"results_path = {results_path}")
        print(f"tmax = {tmax}")
        print(f"tmin = {tmin}")
        print(f"verbose = {verbose}")
        print(f"xmax = {xmax}")
        print(f"xmin = {xmin}")
        print(f"ymax = {ymax}")
        print(f"ymin = {ymin}")

    # If -1 was specified for the model epoch, determine the last epoch.
    if epoch == -1:
        epoch = common.find_last_epoch(results_path)
    if debug:
        print(f"epoch = {epoch}")

    # Load the models for the specified epoch.
    base_path = os.path.join(results_path, "models", str(epoch))
    if debug:
        print(f"base_path = {base_path}")
    path = os.path.join(base_path, "model_Bx")
    if debug:
        print(f"path = {path}")
    model_Bx = tf.keras.models.load_model(path)
    if debug:
        print(f"model_Bx = {model_Bx}")
    path = os.path.join(base_path, "model_By")
    if debug:
        print(f"path = {path}")
    model_By = tf.keras.models.load_model(path)
    if debug:
        print(f"model_By = {model_By}")

    # Create the evaluation grid for the movie frames.
    nxy = nx*ny
    ng = [nt, nx, ny]
    bg = [
        [tmin, tmax],
        [xmin, xmax],
        [ymin, ymax]
    ]
    txy = training_data.create_training_points_gridded(ng, bg)

    # Compute the quiver plot tick locations and labels.
    quiver_x_tick_pos = np.linspace(xmin, xmax, QUIVER_N_X_TICKS)
    quiver_x_tick_labels = [
        f"{(xmin + x/(nx - 1)*(xmax - xmin)):0.1f}" for x in quiver_x_tick_pos
    ]
    quiver_y_tick_pos = np.linspace(ymin, ymax, QUIVER_N_Y_TICKS)
    quiver_y_tick_labels = [
        f"{(ymin + y/(ny - 1)*(ymax - ymin)):0.1f}" for y in quiver_y_tick_pos
    ]

    # Create the frame directory.
    frame_dir = "./frames"
    os.mkdir(frame_dir)

    # Create the frames.
    mpl.use("Agg")
    for i in range(nt):
        i0 = i*nxy
        i1 = i0 + nxy
        if verbose:
            print(f"Creating frame {i} for time {txy[i0, 0]:0.2f}.")
        x = txy[i0:i1, 1]
        y = txy[i0:i1, 2]
        Bx = model_Bx(txy[i0:i1]).numpy()
        By = model_By(txy[i0:i1]).numpy()
        title = f"Magnetic field at t = {txy[i0, 0]:0.2f}"
        mpl.pyplot.clf()
        standard_plots.plot_BxBy_quiver(
            x, y, Bx, By,
            title=title,
            x_tick_pos=quiver_x_tick_pos,
            x_tick_labels=quiver_x_tick_labels,
            y_tick_pos=quiver_y_tick_pos,
            y_tick_labels=quiver_y_tick_labels
        )
        frame_name = f"frame_{i:06d}.png"
        frame_path = os.path.join(frame_dir, frame_name)
        plt.savefig(frame_path)

    # Create the movie.
    os.system(f"convert -delay 10 -loop 0 {frame_dir}/frame_00*.png BxBy.gif")


if __name__ == "__main__":
    """Begin main program."""
    main()
