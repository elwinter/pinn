#!/usr/bin/env python

"""Create data for loop2d_nPuxuyuzBxByBz.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
import problems.loop2d.loop2d_nPuxuyuzBxByBz.loop2d_nPuxuyuzBxByBz as p


# Program constants

# Program description.
description = "Compute initial conditions for loop2d_nPuxuyuzBxByBz problem."

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
    debug = args.debug
    rest = args.rest
    if debug:
        print(f"args = {args}")

    # Fetch the remaining command-line arguments.
    # They should be in 3 sets of 4:
    # t t_min t_max n_t x x_min x_max n_x y y_min y_max n_y
    Xname = rest[::4]
    Xmin = np.array(rest[1::4], dtype=float)
    Xmax = np.array(rest[2::4], dtype=float)
    nX = np.array(rest[3::4], dtype=int)
    assert len(Xname) == len(Xmin) == len(Xmax) == len(nX) == 3
    (tname, xname, yname) = Xname
    (tmin, xmin, ymin) = Xmin
    (tmax, xmax, ymax) = Xmax
    (nt, nx, ny) = nX
    if debug:
        print(f"{tmin} <= t <= {tmax}, nt = {nt}")
        print(f"{xmin} <= x <= {xmax}, nx = {nx}")
        print(f"{ymin} <= y <= {ymax}, ny = {ny}")

    # Create the (t, x, y) grid points for the initial conditions.
    tg = np.linspace(tmin, tmax, nt)
    xg = np.linspace(xmin, xmax, nx)
    yg = np.linspace(ymin, ymax, ny)
    if debug:
        print(f"tg = {tg}")
        print(f"xg = {xg}")
        print(f"yg = {yg}")

    # Print the output header lines.
    header = "# GRID"
    print(header)
    header = "#"
    for (xname, xmin, xmax, nx) in zip(Xname, Xmin, Xmax, nX):
        header += f" {xname} {xmin} {xmax} {nx}"
    print(header)
    header = "#"
    for xname in Xname:
        header += f" {xname}"
    header += " n P ux uy uz Bx By Bz"
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

if __name__ == "__main__":
    """Begin main program."""
    main()
