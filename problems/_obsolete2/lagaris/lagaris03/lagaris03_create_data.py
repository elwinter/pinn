#!/usr/bin/env python

"""Create data for lagaris02 problem.

This is for problem 3 from Lagaris (1998).

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
import problems.lagaris.lagaris03.lagaris03 as p


# Program constants

# Program description.
description = "Create data for lagaris03 problem."


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
    x_min = float(rest[0])
    x_max = float(rest[1])
    n_x = int(rest[2])
    if debug:
        print(f"{x_min} <= x <= {x_max}, n_x = {n_x}")

    # Print the output header lines.
    header = "# GRID"
    print(header)
    header = "# x"
    print(header)
    header = f"# {x_min} {x_max} {n_x}"
    print(header)
    header = "# x Ψ"
    print(header)

    # Compute the data for the boundary condition at x = 0, 1.
    # Each line is:
    # x Ψ
    x = 0.0
    Ψ = p.Ψ_analytical(x)
    print(x, Ψ)
    x = 1.0
    Ψ = p.Ψ_analytical(x)
    print(x, Ψ)


if __name__ == "__main__":
    """Begin main program."""
    main()
