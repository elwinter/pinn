#!/usr/bin/env python

"""Create data for bw1d_nPux.

This problem is a 1-D blast wave, described with n, P, ux.

The blast wave starts at |x| <= R_blast, where P = 1. Everywhere else, P = 0.1.
The fluid is initially motionless (ux = 0) and of uniform density (n = 1.0).
The problem domain is:
    -1 <= x <= 1
    0 <= t <= 1

The initial conditions are:

n = n0 = 1.0
if r <= R_blast:
    P = 10.0
else:
    P = 0.1
ux = 0

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
description = "Compute data for bw1d_nPux problem."

# Constants
P_blast = 10.0  # Blast pressure
R_blast = 0.1   # Radius of initial blast.
n0 = 1.0   # Number density at start
P0 = 0.1   # Pressure at start
u0x = 0.0  # x-component of velocity at start


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
    # They should be in 2 sets of 3:
    # t_min t_max n_t x_min x_max n_x
    assert len(rest) == 6
    (t_min, x_min) = np.array(rest[::3], dtype=float)
    (t_max, x_max) = np.array(rest[1::3], dtype=float)
    (n_t, n_x) = np.array(rest[2::3], dtype=int)
    if debug:
        print(f"{t_min} <= t <= {t_max}, n_t = {n_t}")
        print(f"{x_min} <= x <= {x_max}, n_x = {n_x}")

    # Create the (t, x, y) grid points for the initial conditions.
    tg = np.linspace(t_min, t_max, n_t)
    xg = np.linspace(x_min, x_max, n_x)
    if debug:
        print(f"tg = {tg}")
        print(f"xg = {xg}")

    # Compute the initial conditions at spatial locations.
    # Each line is:
    # tg[0] x n P ux
    for x in xg:
        r = np.sqrt(x**2)
        n = n0
        if r <= R_blast:
            P = P_blast
        else:
            P = P0
        ux = u0x
        print(tg[0], x, n, P, ux)


if __name__ == "__main__":
    """Begin main program."""
    main()
