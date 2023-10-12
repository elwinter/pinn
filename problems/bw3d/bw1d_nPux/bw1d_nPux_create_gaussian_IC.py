#!/usr/bin/env python

"""Create Gaussian initial conditions for the bw1d_nPux problem.

This problem is a 1-D blast wave, described with n, P, ux.

The problem domain is:
    0 <= t <= 1
    -1 <= x <= 1

The initial conditions are:

P = P0 + P_blast*GAUSSIAN(x, mean=0, stddev=0.05)
n = n0*P/P0
ux = u0x

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
n0 = 1.0        # Number density at start
P0 = 0.1        # Pressure at start
P_blast = 1.0   # Peak pressure of blast
stddev_blast = 0.05   # Standard deviation of blast
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
    gaussian_max = norm.pdf(0, loc=0, scale=stddev_blast)
    # for x in xg:
    #     # Gaussian blast of centered at x = 0
    #     P = P0 + P_blast*norm.pdf(x, loc=0, scale=stddev_blast)/gaussian_max
    #     n = n0*P/P0
    #     ux = u0x
    #     print(tg[0], x, n, P, ux)

    # Now add data at the first non-zero time step. Use forward extrapolation
    # from t = 0. At t = 0, dn/dt and dP/dt are 0 by definition, since initial
    # n(x) is flat at n0, and ux(x) is flat at 0. Thus the value of n at the
    # first non-zero time is still n0, and P is still P0, while ux has changed.
    for x in xg:
        # Gaussian blast of centered at x = 0
        P = P0 + P_blast*norm.pdf(x, loc=0, scale=stddev_blast)/gaussian_max
        n = n0*P/P0
        ux = u0x + x/stddev_blast**2*norm.pdf(x, loc=0, scale=stddev_blast)/gaussian_max*(tg[1] - tg[0])
        print(tg[1], x, n, P, ux)


if __name__ == "__main__":
    """Begin main program."""
    main()
