#!/usr/bin/env python

"""Compute initial conditions for square_wave_n.

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
description = "Compute initial conditions for square_wave_n problem."

# Constants
n0 = 0.5   # Ambient number density
n_wave = 2.0   # Square wave number density
x0, x1 = 0.01, 0.21  # Boundaries of initial square wave
u0x = 1.0  # x-component of flow velocity


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
        print(f"args = {args}")

    # Fetch the remaining command-line arguments.
    # They should be in 3 sets of 3:
    # t_min t_max n_t x_min x_max n_x y_min y_max n_y
    assert len(rest) == 6
    (t_min, x_min,) = np.array(rest[::3], dtype=float)
    (t_max, x_max,) = np.array(rest[1::3], dtype=float)
    (n_t, n_x,) = np.array(rest[2::3], dtype=int)
    if debug:
        print(f"{t_min} <= t <= {t_max}, n_t = {n_t}")
        print(f"{x_min} <= x <= {x_max}, n_x = {n_x}")

    # Create the grid points for the initial conditions.
    xg = np.linspace(x_min, x_max, n_x)
    if debug:
        print(f"xg = {xg}")

    # Compute the initial conditions at spatial locations.
    # Each line is:
    # t_min x n
    for x in xg:
        if x0 <= x <= x1:
            print(t_min, x, n_wave)
        else:
            print(t_min, x, n0)


if __name__ == "__main__":
    """Begin main program."""
    main()
