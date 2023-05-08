#!/usr/bin/env python

"""Compute initial conditions for lagaris04.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.


# Program constants

# Program description.
description = "Compute initial conditions for lagaris04 problem."

# Define the problem domain.
x0 = 0.0
x1 = 3.0

# Define the initial condition at x = 0.
Ψ10 = 0.0
Ψ20 = 1.0


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
    return parser


def main():
    """Begin main program."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    debug = args.debug
    if debug:
        print("args = %s" % args)

    # Compute the initial conditions.
    # Each line is:
    # x Ψ10 Ψ20
    print(x0, Ψ10, Ψ20)


if __name__ == "__main__":
    """Begin main program."""
    main()
