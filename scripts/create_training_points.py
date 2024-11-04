#!/usr/bin/env python

"""Create a set of training points.

This program will create a set of training points. The set can be an evenly-
spaced grid in each dimension (the default), or random points in each
dimension.

For gridded results, the user supplies the name, minimum and maximum
values for each dimension, and the number of points in each dimension.

For random points, the user supplies the name, minimum and maximum
values for each dimension, and the total number of points.

The training points are created as a Numpy array, shape (n_train, n_dim),
where n_train is the number of training points, and n_dim is the number of
dimensions in the training space.l

Author
------
Eric Winter (eric.winter62@gmail.com)

"""


# Import standard Python modules.
import argparse
import sys

# Import 3rd-party modules.
import numpy as np

# Import project modules.
from pinn import training_data


# Program constants

# Program description.
description = "Create a set of training points."

# Default random number generator seed.
default_seed = 0


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

    Raises
    ------
    None
    """
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--no0", action="store_true",
        help="Ignore the origin as a data point (default: %(default)s)."
    )
    parser.add_argument(
        "--random", "-r", action="store_true",
        help="Select points randomly within domain (default: %(default)s)."
    )
    parser.add_argument(
        "--seed", type=int, default=default_seed,
        help="Seed for random number generator (default: %(default)s)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    return parser


def main():
    """Begin main program."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}")
    debug = args.debug
    no0 = args.no0
    random = args.random
    seed = args.seed
    verbose = args.verbose
    rest = args.rest

    # Fetch the remaining command-line arguments.
    # For random results:
    # v1name x1min x1max v2name x2min x2max ... n
    # For gridded results:
    # v1name x1min x1max nx1 x2name x2min x2max nx2 ...
    if random:
        Xname = rest[::3]
        Xmin = np.array(rest[1:-1:3], dtype=float)
        Xmax = np.array(rest[2:-1:3], dtype=float)
        n = int(rest[-1])
    else:
        Xname = rest[::4]
        Xmin = np.array(rest[1::4], dtype=float)
        Xmax = np.array(rest[2::4], dtype=float)
        nX = np.array(rest[3::4], dtype=int)
        assert len(Xname) == len(Xmin) == len(Xmax) == len(nX)
    if debug:
        print(f"Xname = {Xname}")
        print(f"Xmin = {Xmin}")
        print(f"Xmax = {Xmax}")
        if random:
            print(f"n = {n}")
        else:
            print(f"nX = {nX}")

    # Assemble the minima and maxima into a combined array of boundaries of
    # the form:
    # [
    #  [x0min, x0max],
    #  [x1min, x1max],
    #  [x2min, x2max],
    #  [x3min, x3max],
    #  ...
    # ]
    b = np.vstack([Xmin, Xmax]).T
    if debug:
        print(f"b = {b}")

    # Create the training points.
    if random:

        # Seed the random number generator.
        np.random.seed(seed)

        # Select the training points randomly within the domain.
        points = training_data.create_training_points_random(n, b)

    else:
        # Create the flattened, evenly-spaced grid. The last dimension varies
        # fastest.
        points = training_data.create_training_points_gridded(nX, b)
    if debug:
        print(f"points = {points}")

    # (Optional) Remove points with a *spatial* coordinate of 0.
    if no0:
        for i in range(len(b) - 1):
            # Remove points where coordinate i is ~0.
            w = np.where(~np.isclose(points[:, i + 1], 0))
            points = points[w]

    # Send the points to standard output.
    # Include a header as a comment describing the data.
    if random:
        header = "# RANDOM"
        print(header)
        header = "#"
        for (xname, xmin, xmax) in zip(Xname, Xmin, Xmax):
            header += f" {xname} {xmin} {xmax}"
        header += f" {n}"
        print(header)
        header = "#"
        for xname in Xname:
            header += f" {xname}"
        print(header)
    else:
        header = "# GRID"
        print(header)
        header = "#"
        for (xname, xmin, xmax, nx) in zip(Xname, Xmin, Xmax, nX):
            header += f" {xname} {xmin} {xmax} {nx}"
        print(header)
        header = "#"
        for xname in Xname:
            header += f" {xname}"
        print(header)
    np.savetxt(sys.stdout, points)


if __name__ == "__main__":
    """Begin main program."""
    main()
