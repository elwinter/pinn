#!/usr/bin/env python

"""Compute initial conditions for static1d.

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
description = "Compute initial conditions for the static1d problem."

# Plasma parameters
n0 = 1.0   # Number density
P0 = 1.0   # Pressure
u0x = 0.0  # x-component of velocity
u0y = 0.0  # y-component of velocity
u0z = 0.0  # z-component of velocity
B0x = 0.0  # x-component of magnetic field
B0y = 0.0  # y-component of magnetic field
B0z = 0.0  # z-component of magnetic field


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
    # They should be in 2 sets of 3:
    # t_min t_max n_t x_min x_max n_x
    assert len(rest) == 6
    (t_min, x_min) = np.array(rest[::3], dtype=float)
    (t_max, x_max) = np.array(rest[1::3], dtype=float)
    (n_t, n_x) = np.array(rest[2::3], dtype=int)
    if debug:
        print("%s <= t <= %s, n_t = %s" % (t_min, t_max, n_t))
        print("%s <= x <= %s, n_x = %s" % (x_min, x_max, n_x))

    # Create the (t, x) grid points for the initial conditions.
    tg = np.linspace(t_min, t_max, n_t)
    xg = np.linspace(x_min, x_max, n_x)
    if debug:
        print("tg = %s" % tg)
        print("xg = %s" % xg)

    # Compute the initial conditions at spatial locations.
    # Each line is:
    # tg[0] x n P ux uy uz Bx By Bz
    for x in xg:
        n = n0
        P = P0
        ux = u0x
        uy = u0y
        uz = u0z
        Bx = B0x
        By = B0y
        Bz = B0z
        print(tg[0], x, n, P, ux, uy, uz, Bx, By, Bz)


if __name__ == "__main__":
    """Begin main program."""
    main()
