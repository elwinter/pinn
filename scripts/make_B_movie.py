#!/usr/bin/env python


"""Create a movie of the magnetic field magnitude as an animated heatmap.

Create a movie of the magnetic field magnitude as an animated heatmap.
The movie is created as an animated GIF. Individual frames are saved as
PNG files in the frames/ subdirectory.
"""


# Import standard modules.
import argparse
# import glob
import os

# # Import 3rd-party modules.
# import matplotlib as mpl
# import numpy as np
# import tensorflow as tf

# Import project-specific modules.
from pinn import common
# from pinn import standard_plots
# from pinn import training_data


# Program constants and defaults

# Program description.
DESCRIPTION = (
    "Create a movie of the magnetic field magnitude as an animated heatmap."
)

# # Default maximum and minimum values for B.
# DEFAULT_BMIN = 1e-4
# DEFAULT_BMAX = 1e-2

# Default epoch to use when selecting a trained model.
# -1 = use last epoch in results.
DEFAULT_EPOCH = -1

# # Default points counts for movie frame generation.
# DEFAULT_NT = 51
# DEFAULT_NX = 51
# DEFAULT_NY = 51

# # Default limits for t, x, y for movie frames.
# DEFAULT_TMIN = 0.0
# DEFAULT_TMAX = 1.0
# DEFAULT_XMIN = -1.0
# DEFAULT_XMAX = 1.0
# DEFAULT_YMIN = -1.0
# DEFAULT_YMAX = 1.0

# # Default path to results directory.
# DEFAULT_RESULTS_PATH = os.getcwd()

# # Tick counts for x- and y-axes in heatmap.
# HEATMAP_N_X_TICKS = 5
# HEATMAP_N_Y_TICKS = 5


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
#     parser.add_argument(
#         "--Bmax", type=float, default=DEFAULT_BMAX,
#         help="Maximum B value to plot (default: %(default)s)."
#     )
#     parser.add_argument(
#         "--Bmin", type=float, default=DEFAULT_BMIN,
#         help="Minimum B value to plot (default: %(default)s)."
#     )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--epoch", "-e", type=int, default=DEFAULT_EPOCH,
        help="Epoch for trained model to use for movie "
             "(default: %(default)s)."
    )
#     parser.add_argument(
#         "--nt", type=int, default=DEFAULT_NT,
#         help="Number of time points to use in movie (default: %(default)s)."
#     )
#     parser.add_argument(
#         "--nx", type=int, default=DEFAULT_NX,
#         help="Number of x points to use in movie (default: %(default)s)."
#     )
#     parser.add_argument(
#         "--ny", type=int, default=DEFAULT_NY,
#         help="Number of y points to use in movie (default: %(default)s)."
#     )
    parser.add_argument(
        "--results_path", "-r", type=str, default=os.getcwd(),
        help="Directory containing model results (default: %(default)s)."
    )
#     parser.add_argument(
#         "--tmax", type=float, default=DEFAULT_TMAX,
#         help="Maximum time value for movie (default: %(default)s)."
#     )
#     parser.add_argument(
#         "--tmin", type=float, default=DEFAULT_TMIN,
#         help="Minimum time value for movie (default: %(default)s)."
#     )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
#     parser.add_argument(
#         "--xmax", type=float, default=DEFAULT_XMAX,
#         help="Maximum x value for movie (default: %(default)s)."
#     )
#     parser.add_argument(
#         "--xmin", type=float, default=DEFAULT_XMIN,
#         help="Minimum x value for movie (default: %(default)s)."
#     )
#     parser.add_argument(
#         "--ymax", type=float, default=DEFAULT_YMAX,
#         help="Maximum y value for movie (default: %(default)s)."
#     )
#     parser.add_argument(
#         "--ymin", type=float, default=DEFAULT_YMIN,
#         help="Minimum y value for movie (default: %(default)s)."
#     )
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
#     Bmax = args.Bmax
#     Bmin = args.Bmin
    debug = args.debug
    epoch = args.epoch
#     nt = args.nt
#     nx = args.nx
#     ny = args.ny
    results_path = args.results_path
#     tmax = args.tmax
#     tmin = args.tmin
    verbose = args.verbose
#     xmax = args.xmax
#     xmin = args.xmin
#     ymax = args.ymax
#     ymin = args.ymin
    if debug:
        print(f"args = {args}")
#         print(f"Bmax = {Bmax}")
#         print(f"Bmin = {Bmin}")
        print(f"debug = {debug}")
        print(f"epoch = {epoch}")
#         print(f"nt = {nt}")
#         print(f"nx = {nx}")
#         print(f"ny = {ny}")
        print(f"results_path = {results_path}")
#         print(f"tmax = {tmax}")
#         print(f"tmin = {tmin}")
        print(f"verbose = {verbose}")
#         print(f"xmax = {xmax}")
#         print(f"xmin = {xmin}")
#         print(f"ymax = {ymax}")
#         print(f"ymin = {ymin}")

    # If -1 was specified for the model epoch, determine the last epoch.
    if epoch == -1:
        epoch = common.find_last_epoch(results_path)
    if debug:
        print(f"epoch = {epoch}")

#     # Load the models for the specified epoch.
#     base_path = os.path.join(results_path, "models", str(epoch))
#     path = os.path.join(base_path, "model_Bx")
#     model_Bx = tf.keras.models.load_model(path)
#     path = os.path.join(base_path, "model_By")
#     model_By = tf.keras.models.load_model(path)

#     # Create the evaluation grid for the movie frames.
#     nxy = nx*ny
#     ng = [nt, nx, ny]
#     bg = [
#         [tmin, tmax],
#         [xmin, xmax],
#         [ymin, ymax]
#     ]
#     txy = training_data.create_training_points_gridded(ng, bg)

#     # Compute the heat map tick locations and labels.
#     heatmap_x_tick_pos = np.linspace(0, nx - 1, HEATMAP_N_X_TICKS)
#     heatmap_x_tick_labels = [
#         f"{(xmin + x/(nx - 1)*(xmax - xmin)):0.1f}" for x in heatmap_x_tick_pos
#     ]
#     heatmap_y_tick_pos = np.linspace(0, ny - 1, HEATMAP_N_Y_TICKS)
#     heatmap_y_tick_labels = [
#         f"{(ymin + y/(ny - 1)*(ymax - ymin)):0.1f}" for y in heatmap_y_tick_pos
#     ]
#     heatmap_y_tick_labels = list(reversed(heatmap_y_tick_labels))

#     # Create the frame directory.
#     frame_dir = "./frames"
#     os.mkdir(frame_dir)

#     # Create the frames.
#     mpl.use("Agg")
#     for i in range(nt):
#         i0 = i*nxy
#         i1 = i0 + nxy
#         if verbose:
#             print(f"Creating frame {i} for time {txy[i0, 0]:0.2f}.")
#         Bx = model_Bx(txy[i0:i1]).numpy()
#         By = model_By(txy[i0:i1]).numpy()
#         B = np.sqrt(Bx**2 + By**2)
#         B = np.flip(B.reshape(nx, ny).T, axis=0)
#         title = f"Magnetic field magnitude at t = {txy[i0, 0]:0.2f}"
#         mpl.pyplot.clf()
#         standard_plots.plot_logarithmic_heatmap(
#             B, title=title,
#             vmin=Bmin, vmax=Bmax,
#             x_tick_pos=heatmap_x_tick_pos,
#             x_tick_labels=heatmap_x_tick_labels,
#             y_tick_pos=heatmap_y_tick_pos,
#             y_tick_labels=heatmap_y_tick_labels
#         )
#         frame_name = f"frame_{i:06d}.png"
#         frame_path = os.path.join(frame_dir, frame_name)
#         mpl.pyplot.savefig(frame_path)

#     # Create the movie.
#     os.system(f"convert -delay 10 -loop 0 {frame_dir}/frame_00*.png B.gif")


if __name__ == "__main__":
    """Begin main program."""
    main()
