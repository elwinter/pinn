#!/usr/bin/env python

"""Create hat-function initial conditions for the bw1d_nPux problem.

This problem is a 1-D blast wave, described with n, P, ux.

The problem domain is:
    0 <= t <= 1
    -1 <= x <= 1

The initial conditions are:

if x <= R_blast:
    n = n_blast
    P = P_blast
else:
    n = n0
    P = P0
ux = ux0

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
description = "Create hat-function initial conditions for the bw1d_nPux problem."

# Constants

# Ambient initial conditions
n0 = 0.1
P0 = 0.1
u0x = 0.0

# Blast parameters
# Ideal, isothermal gas: P = n*k*T
P_blast = 1.0   # Blast pressure
n_blast = 1.0   # Blast number density
R_blast = 0.1   # Blast radius


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
    for x in xg:
        if np.abs(x) <= R_blast:
            n = n_blast
            P = P_blast
        else:
            n = n0
            P = P0
        ux = u0x
        print(tg[0], x, n, P, ux)


if __name__ == "__main__":
    """Begin main program."""
    main()
