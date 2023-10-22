#!/usr/bin/env python

"""Create Gaussian adiabatic initial conditions for the bw1d_nPux problem.

This problem is an *adiabatic* 1-D blast wave, described with n, P, ux.

The problem domain is:
    0 <= t <= 1
    0 <= x <= 1

The initial conditions are:

P = P0 + P1*GAUSSIAN(x, mean=0, stddev=0.05)
n = n0*(P0/P)**gamma
ux = u0x + u1x*GAUSSIAN(x, mean=0, stddev=0.05)

where GAUSSIAN(x, mu, stddev) is the Gaussian distribution, normalized to unit
area, evaluated at x, with mean mu and standard deviation stddev.

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
description = "Create Gaussian adiabatic data for the bw1d_nPux problem."

# Constants
n0 = 1.0        # Number density at start
P0 = 0.1        # Pressure at start
u0x = 0.0       # x-component of velocity at start
gamma = 5/3     # Adiabatic index of gas

# Blast characteristics
P1 = 1.0              # Additional peak pressure of blast
stddev_blast = 0.05   # Standard deviation in x of blast pressure at start
                      # Also used as stddev for ux(0, x) distribution.
u1x = 1.0             # x-component of blast velocity at start


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

    # Write the training data header.
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

    # gaussian_max is the value of the Gaussian of the specified standard
    # deviation, at the center of the distribution (the origin). Dividing by
    # this value normalizes the Gaussian distribution to unit area.
    gaussian_max = norm.pdf(0, loc=0, scale=stddev_blast)

    # Compute the initial conditions.
    # Assume a Gaussian P distribution centered at the origin.
    # n is computed from P using the adiabatic relation.
    # ux is arbitrary. so set to Gaussian centered at origin, using the same
    # standard deviation (this is not a requirement, just a convenience).
    for x in xg:
        P = P0 + P1*norm.pdf(x, loc=0, scale=stddev_blast)/gaussian_max
        n = n0*(P0/P)**gamma
        ux = u0x + u1x*norm.pdf(x, loc=0, scale=stddev_blast)/gaussian_max
        print(tg[0], x, n, P, ux)

    # Now add the boundary conditions at x = 0.
    # n, P, ux are all constant at their initial values.
    for t in tg:
        # Gaussian blast of centered at x = 0
        P = P0 + P1
        n = n0*(P0/P)**gamma
        ux = u0x + u1x
        print(t, 0, n, P, ux)


if __name__ == "__main__":
    """Begin main program."""
    main()
