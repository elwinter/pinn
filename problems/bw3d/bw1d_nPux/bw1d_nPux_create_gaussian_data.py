#!/usr/bin/env python

"""Compute Gaussian initial conditions for bw1d_nPux problem.

This problem is a 1-D blast wave, described with n, P, ux.

The problem domain is:
    -1 <= x <= 1
    0 <= t <= 1

The initial conditions are:

n = n0 = 1.0
P = E_blast*GAUSSIAN(mean=0, stddev=0.05)
ux = 0

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
description = "Compute Gaussian initial conditions for bw1d_nPux problem."

# Constants
n0 = 1.0        # Number density at start
P0 = 0.1        # Pressure at start
P_blast = 1.0   # Pressure of equivalent hat-function blast
R_blast = 0.1   # Radius of initial blast.
E_blast = P_blast*2*R_blast  # Total energy of hat-function blast
u0x = 0.0       # x-component of velocity at start


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

    # First 2 lines are comment header.
    # Each subsequent line is:
    # tg[0] x n P ux
    header = "# t x"
    print(header)
    header = f"# {t_min} {t_max} {n_t} {x_min} {x_max} {n_x}"
    print(header)
    header = "# t x n P ux"
    print(header)
    for x in xg:
        r = np.sqrt(x**2)
        n = n0
        # Gaussian blast of same total energy as step function
        # Centered at x = 0, with stddev = 0.05, so blast mostly
        # contained inside |x| < 0.25.
        P = P0 + E_blast*norm.pdf(x, loc=0, scale=0.05)
        ux = u0x
        print(tg[0], x, n, P, ux)


if __name__ == "__main__":
    """Begin main program."""
    main()
