#!/usr/bin/env python

"""Compute initial conditions for bw1d_nPuxuyuzBxByBz.

This problem is a 1-D blast wave, described with n, P, ux, uy, uz, Bx, By, Bz.

The problem domain is:
    -1 <= x <= 1
    0 <= t <= 1

The initial conditions are a hat function in pressure:

n = 1.0
if r <= R_blast:
    P = 10.0
else:
    P = 0.1
ux = 0
uy = 0
uz = 0
Bx = 0
By = 0
Bz = 0

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
description = "Compute initial conditions for bw1d_nPuxuyuzBxByBz problem."

# Constants
n0 = 1.0        # Number density at start
P0 = 0.1        # Pressure at start
P_blast = 1.0   # Blast pressure
R_blast = 0.1   # Radius of initial blast.
u0x = 0.0       # x-component of velocity at start
u0y = 0.0       # y-component of velocity start
u0z = 0.0       # z-component of velocity start
B0x = 0.0       # x-component of magnetic field at start
B0y = 0.0       # y-component of magnetic field at start
B0z = 0.0       # z-component of magnetic field at start


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
    # They should be in 2 sets of 3:
    # t_min t_max n_t x_min x_max n_x
    assert len(rest) == 6
    (t_min, x_min) = np.array(rest[::3], dtype=float)
    (t_max, x_max) = np.array(rest[1::3], dtype=float)
    (n_t, n_x) = np.array(rest[2::3], dtype=int)
    if debug:
        print(f"{t_min} <= t <= {t_max}, n_t = {n_t}")
        print(f"{x_min} <= x <= {x_max}, n_x = {n_x}")

    # Create the (t, x) grid points for the initial conditions.
    tg = np.linspace(t_min, t_max, n_t)
    xg = np.linspace(x_min, x_max, n_x)
    if debug:
        print(f"tg = {tg}")
        print(f"xg = {xg}")

    # Compute the initial conditions at spatial locations.
    # First 3 lines are comment header.
    # Each subsequent line is:
    # tg[0] x n P ux uy uz Bx By Bz
    header = "# t x"
    print(header)
    header = f"# {t_min} {t_max} {n_t} {x_min} {x_max} {n_x}"
    print(header)
    header = "# t x n P ux uy uz Bx By Bz"
    print(header)
    for x in xg:
        r = np.sqrt(x**2)
        n = n0
        if r <= R_blast:
            P = P_blast
        else:
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
