#!/usr/bin/env python

"""Compute initial conditions for static3d.

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
description = "Compute initial conditions for the static3d problem."

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
        print(f"args = {args}")

    # Fetch the remaining command-line arguments.
    # They should be in 4 sets of 3:
    # t_min t_max n_t x_min x_max n_x y_min y_max n_y z_min z_max n_z
    assert len(rest) == 12
    (t_min, x_min, y_min, z_min) = np.array(rest[::3], dtype=float)
    (t_max, x_max, y_max, z_max) = np.array(rest[1::3], dtype=float)
    (n_t, n_x, n_y, n_z) = np.array(rest[2::3], dtype=int)
    if debug:
        print(f"{t_min} <= t <= {t_max}, n_t = {n_t}")
        print(f"{x_min} <= x <= {x_max}, n_x = {n_x}")
        print(f"{y_min} <= y <= {y_max}, n_y = {n_y}")
        print(f"{z_min} <= z <= {z_max}, n_z = {n_z}")

    # Create the (t, x, y, z) grid points for the initial conditions.
    tg = np.linspace(t_min, t_max, n_t)
    xg = np.linspace(x_min, x_max, n_x)
    yg = np.linspace(y_min, y_max, n_y)
    zg = np.linspace(z_min, z_max, n_z)
    if debug:
        print(f"tg = {tg}")
        print(f"xg = {xg}")
        print(f"yg = {yg}")
        print(f"zg = {zg}")

    # Compute the initial conditions at spatial locations.
    # Each line is:
    # tg[0] x y z n P ux uy uz Bx By Bz
    for x in xg:
        for y in yg:
            for z in zg:
                n = n0
                P = P0
                ux = u0x
                uy = u0y
                uz = u0z
                Bx = B0x
                By = B0y
                Bz = B0z
                print(tg[0], x, y, z, n, P, ux, uy, uz, Bx, By, Bz)


if __name__ == "__main__":
    """Begin main program."""
    main()
