#!/usr/bin/env python

"""Compute data for lagaris01.

This is for problem 1 from Lagaris (1998).

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
import problems.lagaris.lagaris01.lagaris01 as p


# Program constants

# Program description.
description = "Compute data for lagaris01 problem."

# # Define the problem domain.
# x0 = 0.0
# x1 = 1.0

# # Define the initial condition at x = 0.
# Ψ0 = 1.0


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
    """
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        "-d", "--debug", action="store_true",
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
    debug = args.debug
    rest = args.rest
    if debug:
        print("args = %s" % args)

    # Fetch the remaining command-line arguments.
    # They should be in a set of 3:
    # x_min x_max n_x
    assert len(rest) == 3
    x_min = float(rest[0])
    x_max = float(rest[1])
    n_x = int(rest[2])
    if debug:
        print(f"{x_min} <= x <= {x_max}, n_x = {n_x}")

    # Compute the data.
    # Each line is:
    # x Ψ
    x_data = np.linspace(x_min, x_max, n_x)
    if debug:
        print(f"x_data = {x_data}")
    Ψ_data = p.Ψ_analytical(x_data).numpy()
    if debug:
        print(f"Ψ_data = {Ψ_data}")
    for (x, Ψ) in zip(x_data, Ψ_data):
        print(x, Ψ)


if __name__ == "__main__":
    """Begin main program."""
    main()
