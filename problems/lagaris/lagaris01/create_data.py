#!/usr/bin/env python

"""Create data for the lagaris01 problem.

Create data for the lagaris01 problem.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
from pinn import common
import problems.lagaris.lagaris01.lagaris01 as p


# Program constants

# Program description.
DESCRIPTION = "Create data for the lagaris01 problem."


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
    # Create the minimal command-line parser.
    parser = common.create_minimal_command_line_argument_parser(DESCRIPTION)

    # Add arguments specific to this script.
    parser.add_argument(
        "--file", "-f", default=None,
        help="Path to file of input points (default: %(default)s)"
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER)

    # Return the parser.
    return parser


def create_data(args: dict):
    """Create data for the lagaris01 problem.

    Create data for the lagaris01 problem.

    Parameters
    ----------
    args: dict
        Dictionary of command-line options.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Local convenience variables
    debug = args["debug"]
    _file = args["file"]
    rest = args["rest"]

    # If the evaluation points are provided in a file, read them. Otherwise,
    # read the point limits from the rest of the command line.
    # NOTE: These options are mutually exclusive.
    if _file:
        if len(rest) > 0:
            raise TypeError("Must not specify --file and point limits!")
        X = np.loadtxt(_file)
    else:

        # Extract grid parameters.
        # They should be in a set of 3 for each independent variable:
        # min max n
        xmin = float(rest[0])
        xmax = float(rest[1])
        nx = int(rest[2])
        if debug:
            print(f"{xmin} <= x <= {xmax}, nx = {nx}")

        # Compute the range of independent variables to use.
        X = np.linspace(xmin, xmax, nx)

    if debug:
        print(f"X = {X}")

    # Compute each dependent variable at each point.
    Y = p.Y_analytical[0](X)
    if debug:
        print(f"Y = {Y}")

    # Print the output header lines.
    header = "# GRID"
    print(header)
    header = "# x"
    print(header)
    header = "# x Ψ"
    print(header)

    # Print each data point.
    # Each line is:
    # x Ψ
    for (x, y) in zip(X, Y):
        print(x, y)


def main():
    """Driver for command-line version of code."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}")

    # Pass the command-line arguments to the main function as a dict.
    args = vars(args)
    create_data(args)


if __name__ == "__main__":
    main()
