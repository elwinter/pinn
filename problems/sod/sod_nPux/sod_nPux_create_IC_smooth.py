#!/usr/bin/env python

"""Create initial conditions for the Sod shock tube problem.

This problem is described at:

http://wonka.physics.ncsu.edu/pub/VH-1/bproblems.php

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
description = "Create Gaussian data for the bw1d_nPux problem."

# Constants
nl, nr = 1.0, 0.125   # Number density on left and right of shock at start
dn = nr - nl          # Change across discontinuity
Pl, Pr = 1.0, 0.1     # Pressure on left and right of shock at start
dP = Pr - Pl          # Change across discontinuity

# Constants for the transition from left to right
k = 40.0


def s(x, k):
    return 1/(1 + np.exp(-k*x))


def ds_dx(x, k):
    _s = s(x, k)
    return k*_s*(1 - _s)


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
    # There should be 2 sets of 3:
    # t_min t_max n_t x_min x_max n_x
    assert len(rest) == 6
    (t_min, x_min) = np.array(rest[::3], dtype=float)
    (t_max, x_max) = np.array(rest[1::3], dtype=float)
    (n_t, n_x) = np.array(rest[2::3], dtype=int)
    if debug:
        print(f"{t_min} <= t <= {t_max}, n_t = {n_t}")
        print(f"{x_min} <= x <= {x_max}, n_x = {n_x}")

    # Create the (t, x) grid points for the data.
    tg = np.linspace(t_min, t_max, n_t)
    xg = np.linspace(x_min, x_max, n_x)
    if debug:
        print(f"tg = {tg}")
        print(f"xg = {xg}")

    # Compute the data at each point.
    # First 4 lines are metadata header as comments.
    # Each subsequent line is:
    # t x n P ux
    header = "# GRID"
    print(header)
    header = "# t x"
    print(header)
    header = f"# {t_min} {t_max} {n_t} {x_min} {x_max} {n_x}"
    print(header)
    header = "# t x n P ux"
    print(header)
    # Smooth transition around x = 0.5
    xmid = 0.5
    for x in xg:
        n = nl + dn*s(x - xmid, k)
        P = Pl + dP*s(x - xmid, k)
        dP_dx = dP*k*ds_dx(x - xmid, k)
        ux = -dP_dx/n
        print(tg[0], x, n, P, ux)


if __name__ == "__main__":
    """Begin main program."""
    main()
