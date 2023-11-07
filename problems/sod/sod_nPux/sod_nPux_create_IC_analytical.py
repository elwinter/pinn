#!/usr/bin/env python

"""Create analytical initial conditions for the Sod shock tube problem.

This problem is described at:

http://wonka.physics.ncsu.edu/pub/VH-1/bproblems.php

The code for computing the analytical solution is from:

https://github.com/ibackus/sod-shocktube

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np
from scipy.stats import norm
import sodshock

# Import project Python modules.


# Program constants

# Program description.
description = "Create analytical initial conditions for the Sod shock tube problem."

# Constants
nl, nr = 1.0, 0.125   # Number density on left and right of shock at start
Pl, Pr = 1.0, 0.1     # Pressure on left and right of shock at start
gamma = 1.4


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

    # Compute the analytical solution at the time resolution of the grid.
    # Use the solution at t = tg[1] as the initial condition.
    x_shock = 0.5
    positions, regions, values = sodshock.solve(
        left_state=[Pl, nl, 0.0],
        right_state=[Pr, nr, 0.0],
        geometry=(x_min, x_max, x_shock), t=tg[1], 
        gamma=gamma, npts=n_x
    )

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
    for (x, n, P, ux) in zip(values["x"], values["rho"], values["p"], values["u"]):
        print(tg[1], x, n, P, ux)


if __name__ == "__main__":
    """Begin main program."""
    main()
