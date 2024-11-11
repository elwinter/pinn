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
import copy
import sys

# Import 3rd-party modules.
import numpy as np

# Import project modules.
from pinn import common
from pinn import training_data


# Program constants

# Program description.
DESCRIPTION = "Create a set of training points."

# Default values for command-line arguments when none are supplied (such as
# when create_training_points() is called by external code).
args_default = {
    "debug": False,
    "no0": False,
    "problem_path": None,
    "randomize": False,
    "seed": 0,
    "verbose": False,
}

def create_command_line_parser(description: str = DESCRIPTION):
    """Create the command-line parser.

    Create the command-line parser.

    Parameters
    ----------
    description : str, default DESCRIPTION
        Description string for script.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser for the command-line.

    Raises
    ------
    None
    """
    parser = common.create_minimal_command_line_parser(description)
    parser.add_argument(
        "--no0", action="store_true",
        help="Ignore the origin as a data point (default: %(default)s)."
    )
    parser.add_argument(
        "--problem", type=str, default=args_default["problem_path"],
        help="If set, compute dependent variable values using the analytical "
        "solutions defined in this Python file (default: %(default)s)."
    )
    parser.add_argument(
        "--randomize", "-r", action="store_true",
        help="Select points randomly within domain (default: %(default)s)."
    )
    parser.add_argument(
        "--seed", type=int, default=args_default["seed"],
        help="Seed for random number generator (default: %(default)s)"
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    return parser


def create_training_points(args: dict):
    """Create training points.

    Create training points.

    Parameters
    ----------
    args : dict
        Dictionary of command-line and other options.

    Returns
    -------
    header : list of str
        Header lines describing data.
    data : np.ndarray of object
        Array of data.

    Raises
    ------
    None
    """
    # Use defaults for unspecified arguments. Merge additional settings
    # which are passed in by the caller.
    local_args = copy.deepcopy(args_default)
    if args is not None:
        local_args.update(args)
    args = local_args
    if args["debug"]:
        print(f"args = {args}")

    # Local convenience variables.
    debug = args["debug"]
    no0 = args["no0"]
    problem = args["problem"]
    randomize = args["randomize"]
    seed = args["seed"]
    verbose = args["verbose"]
    rest = args["rest"]

    # ------------------------------------------------------------------------

    # Fetch the remaining command-line arguments.
    if randomize:
        # For random points:
        # x1name x1min x1max x2name x2min x2max ... n
        xname = rest[:-1:3]
        xmin = np.array(rest[1:-1:3], dtype=float)
        xmax = np.array(rest[2:-1:3], dtype=float)
        nr = int(rest[-1])
        if debug:
            for i in range(len(xname)):
                print(f"{xmin[i]} <= {xname[i]} <= {xmax[i]}")
            print(f"nr = {nr}")
    else:
        # For gridded points:
        # x1name x1min x1max nx1 x2name x2min x2max nx2 ...
        xname = rest[::4]
        xmin = np.array(rest[1::4], dtype=float)
        xmax = np.array(rest[2::4], dtype=float)
        nx = np.array(rest[3::4], dtype=int)
        if debug:
            for i in range(len(xname)):
                print(f"{xmin[i]} <= {xname[i]} <= {xmax[i]}, nx = {nx[i]}")

    # Create a list for the header lines.
    header = []
    if randomize:
        line = "# RANDOM"
        for i in range(len(xname)):
            line += f" {xname[i]} {xmin[i]} {xmax[i]}"
        line += f" {nr}"
        header.append(line)
    else:
        line = "# GRID"
        for i in range(len(xname)):
            line += f" {xname[i]} {xmin[i]} {xmax[i]} {nx[i]}"
        header.append(line)
    header.append(f"# {' '.join(xname)}")

    # Assemble the minima and maxima into a combined array of boundaries of
    # the form:
    # [
    #  [x0min, x0max],
    #  [x1min, x1max],
    #  [x3min, x3max],
    #  ...
    # ]
    b = np.vstack([xmin, xmax]).T
    if debug:
        print(f"b = {b}")

    # Compute the value at each grid point.
    # Each point is a row of the form:
    # x0 x1 x2 ...
    if randomize:
        np.random.seed(seed)
        x = training_data.create_training_points_random(nr, b)
    else:
        x = training_data.create_training_points_gridded(nx, b)

    # Remove points with a coordinate of 0.
    if no0:
        for i in range(len(b)):
            w = np.where(~np.isclose(x[:, i], 0.0))
            x = x[w]


    # If a problem was specified, compute the value of each
    # solution function at each point, then augment the array with
    # a column for each dependent variable.
    xy = None
    if problem:
        p = common.import_problem(problem)
        y = np.hstack([f(x) for f in p.Y_analytical])
        xy = np.hstack([x, y])
        # Add names for new columns.
        header[1] += f" {' '.join(p.dependent_variable_names)}"
    else:
        xy = x

    # Return the header and data.
    return header, xy


def main():
    """Top-level code for the command-line version of create_training_points.

    This is the top-level code for the command-line version of
    create_training_points. It processes command-line options, then calls the
    create_training_points() function. The results are then printed to stdout.

    Parameters
    ----------
    None

    Returns
    -------
    0 on success

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

    # Call the main program logic.
    header, data = create_training_points(args)

    # Print the data.
    for line in header:
        print(line)
    for x in data:
        print(f"{' '.join([str(_x) for _x in x])}")

    # Exit normally.
    sys.exit(0)


if __name__ == "__main__":
    main()
