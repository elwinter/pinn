#!/usr/bin/env python

"""Create data for eplasma1.

The data are the perturbation values for number density, x-velocity, and
x-electric field for (t, x) = (0, x) (the initial conditions) and
(t, x) = (t, 0) (the time-dependent boundary conditions at x=0). Now includes
data at end time.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
from pinn import plasma
import problems.waves1d.eplasma1.eplasma1 as p


# Program constants

# Program description.
description = "Create data for eplasma1 problem."


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
    (t_min, x_min,) = np.array(rest[::3], dtype=float)
    (t_max, x_max,) = np.array(rest[1::3], dtype=float)
    (n_t, n_x,) = np.array(rest[2::3], dtype=int)
    if debug:
        print(f"{t_min} <= t <= {t_max}, n_t = {n_t}")
        print(f"{x_min} <= x <= {x_max}, n_x = {n_x}")

    # Create the (t, x) grid points for the data.
    tg = np.linspace(t_min, t_max, n_t)
    xg = np.linspace(x_min, x_max, n_x)
    if debug:
        print(f"tg = {tg}")
        print(f"xg = {xg}")

    # First 4 lines are metadata header as comments.
    # Each subsequent line is:
    # t x n1 u1x E1x
    header = "# GRID"
    print(header)
    header = "# t x"
    print(header)
    header = f"# {t_min} {t_max} {n_t} {x_min} {x_max} {n_x}"
    print(header)
    header = "# t x n P ux"
    print(header)

    # Compute the initial conditions at (t=0, x).
    # Each line is:
    # tg[0] x n1 u1x E1x
    tx = np.zeros((n_x, 2))
    tx[:, 0] = tg[0]
    tx[:, 1] = xg
    n1 = p.n1_analytical(tx)
    u1x = p.u1x_analytical(tx)
    E1x = p.E1x_analytical(tx)
    for i in range(n_x):
        print(tx[i, 0], tx[i, 1], n1[i], u1x[i], E1x[i])

    # Compute the boundary conditions at (t, x=0).
    # Each line is:
    # t xg[0] n1 u1x E1x
    tx = np.zeros((n_t, 2))
    tx[:, 0] = tg
    tx[:, 1] = xg[0]
    n1 = p.n1_analytical(tx)
    u1x = p.u1x_analytical(tx)
    E1x = p.E1x_analytical(tx)
    for i in range(n_t):
        print(tx[i, 0], tx[i, 1], n1[i], u1x[i], E1x[i])

    # Compute the initial conditions at (t=end, x).
    # Each line is:
    # tg[-1] x n1 u1x E1x
    tx = np.zeros((n_x, 2))
    tx[:, 0] = tg[-1]
    tx[:, 1] = xg
    n1 = p.n1_analytical(tx)
    u1x = p.u1x_analytical(tx)
    E1x = p.E1x_analytical(tx)
    for i in range(n_x):
        print(tx[i, 0], tx[i, 1], n1[i], u1x[i], E1x[i])


if __name__ == "__main__":
    """Begin main program."""
    main()
