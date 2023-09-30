#!/usr/bin/env python

"""Compute initial conditions for bw2d_nPuxuyuzBxByBz.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np
from scipy.stats import norm

# Import project Python modules.


# Program constants

# Program description.
description = "Compute initial conditions for bw2d_nPuxuyuzBxByBz problem."

# Constants
P_blast = 1.0
R0 = 0.1   # Radius of initial blast.
E_blast = P_blast*2*R0
blast_mean = 0
blast_stddev = 0.05

n0 = 1.0   # Number density at start
P0 = 0.1   # Pressure at start

# Compute the constant magnetic field components.
B0x = 0.0  # x-component of magnetic field at start
B0y = 0.0  # y-component of magnetic field at start
B0z = 0.0  # z-component of magnetic field at start

# Compute the constant velocity components.
u0x = 0.0  # x-component of velocity at start
u0y = 0.0  # y-component of velocity at start
u0z = 0.0  # z-component of velocity at start


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
    # They should be in 4 sets of 3:
    # t_min t_max n_t x_min x_max n_x y_min y_max n_y
    assert len(rest) == 9
    (t_min, x_min, y_min) = np.array(rest[::3], dtype=float)
    (t_max, x_max, y_max) = np.array(rest[1::3], dtype=float)
    (n_t, n_x, n_y) = np.array(rest[2::3], dtype=int)
    if debug:
        print(f"{t_min} <= t <= {t_max}, n_t = {n_t}")
        print(f"{x_min} <= x <= {x_max}, n_x = {n_x}")
        print(f"{y_min} <= y <= {y_max}, n_y = {n_y}")

    # Create the (t, x, y) grid points for the initial conditions.
    tg = np.linspace(t_min, t_max, n_t)
    xg = np.linspace(x_min, x_max, n_x)
    yg = np.linspace(y_min, y_max, n_y)
    if debug:
        print(f"tg = {tg}")
        print(f"xg = {xg}")
        print(f"yg = {yg}")

    # Compute the initial conditions at spatial locations.
    # Each line is:
    # tg[0] x y n P ux uy uz Bx By Bz
    for x in xg:
        for y in yg:
            r = np.sqrt(x**2 + y**2)
            n = n0
            P = E_blast*norm.pdf(r
            ux = u0x
            uy = u0y
            uz = u0z
            Bx = B0x
            By = B0y
            Bz = B0z
            print(tg[0], x, y, n, P, ux, uy, uz, Bx, By, Bz)


if __name__ == "__main__":
    """Begin main program."""
    main()
