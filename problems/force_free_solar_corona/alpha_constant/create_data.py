#!/usr/bin/env python

"""Create data for the alpha_constant problem.

Create data for the alpha_constant problem.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
from pinn import common
import problems.force_free_solar_corona.alpha_constant.alpha_constant as p


# Program constants

# Program description.
DESCRIPTION = "Create data for the alpha_constant problem."


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
    # Create the minimal command-line parser.
    parser = common.create_minimal_command_line_argument_parser(DESCRIPTION)

    # Add arguments specific to this script.
    parser.add_argument(
        "--file", "-f", default=None,
        help="Path to file of input points (default: %(default)s)"
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER)

    # Return the parser.
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
        print("args = %s" % args)

    # Fetch the remaining command-line arguments.
    # They should be in 3 sets of 3:
    # t_min t_max n_t x_min x_max n_x y_min y_max n_y
    assert len(rest) == 9
    (t_min, x_min, y_min) = np.array(rest[::3], dtype=float)
    (t_max, x_max, y_max) = np.array(rest[1::3], dtype=float)
    (n_t, n_x, n_y) = np.array(rest[2::3], dtype=int)
    if debug:
        print("%s <= t <= %s, n_t = %s" % (t_min, t_max, n_t))
        print("%s <= x <= %s, n_x = %s" % (x_min, x_max, n_x))
        print("%s <= y <= %s, n_y = %s" % (y_min, y_max, n_y))

    # Create the (t, x, y) grid points for the initial conditions.
    tg = np.linspace(t_min, t_max, n_t)
    xg = np.linspace(x_min, x_max, n_x)
    yg = np.linspace(y_min, y_max, n_y)
    if debug:
        print("tg = %s" % tg)
        print("xg = %s" % xg)
        print("yg = %s" % yg)

    # Print the output header lines.
    header = "# GRID"
    print(header)
    header = "# t x y"
    print(header)
    header = f"# {t_min} {t_max} {n_t} {x_min} {x_max} {n_x} {y_min} {y_max} {n_y}"
    print(header)
    header = "# t x y n P ux uy uz Bx By Bz"
    print(header)

    # Compute the initial conditions at spatial locations.
    # Each line is:
    # tg[0] x y n P ux uy uz Bx By Bz
    t0 = np.array([tg[0]])
    for x in xg:
        for y in yg:
            r = np.sqrt(x**2 + y**2)
            n = p.n0
            P = p.P0
            ux = p.u0x
            uy = p.u0y
            uz = p.u0z
            if r < p.R0:
                Bx = -p.A*y/r
                By = p.A*x/r
            else:
                Bx = p.B0x
                By = p.B0y
            Bz = p.B0z
            print(tg[0], x, y, n, P, ux, uy, uz, Bx, By, Bz)


def main():
    """Driver for command-line version of code."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}")

    # Pass the command-line arguments to the main function as a dict.
    args = vars(args)
    create_data(args)


if __name__ == "__main__":
    main()
