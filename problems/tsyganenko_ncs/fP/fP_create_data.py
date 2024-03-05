#!/usr/bin/env python

"""Create data for fP problem.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
import problems.tsyganenko_ncs.fP.fP as p


# Program constants

# Program description.
description = "Create data for fP problem."


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
    rest = args.rest

    # Fetch the remaining command-line arguments.
    # They should be in a set of 3:
    # x_min x_max n_x
    assert len(rest) == 3
    xmin = float(rest[0])
    xmax = float(rest[1])
    nx = int(rest[2])
    if debug:
        print(f"{xmin} <= x <= {xmax}, nx = {nx}")

    # Print the output header lines.
    header = "# GRID"
    print(header)
    header = "# P"
    print(header)
    header = f"# {xmin} {xmax} {nx}"
    print(header)
    header = "# P fP"
    print(header)

    # Compute the data for the boundary condition at x = 0.
    # Each line is:
    # P fP
    P = np.linspace(xmin, xmax, nx)
    fP = p.fP_analytical(P)
    for P, fP in zip(P, fP):
        print(f"{P} {fP}")


if __name__ == "__main__":
    """Begin main program."""
    main()
