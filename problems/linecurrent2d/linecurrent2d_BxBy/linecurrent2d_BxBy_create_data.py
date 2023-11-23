#!/usr/bin/env python

"""Create data for linecurrent_BxBy.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
import problems.linecurrent2d.linecurrent2d_BxBy.linecurrent2d_BxBy as p


# Program constants

# Program description.
description = "Create data for linecurrent2d_BxBy problem."


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
    header = "# t x y Bx By"
    print(header)

    # Compute the initial conditions at spatial locations.
    # Each line is:
    # tg[0] x y Bx By
    for x in xg:
        for y in yg:
            r = np.sqrt(x**2 + y**2)
            Bx = p.Bx_analytical(tg[0], x, y)
            By = p.By_analytical(tg[0], x, y)
            print(tg[0], x, y, Bx, By)


if __name__ == "__main__":
    """Begin main program."""
    main()