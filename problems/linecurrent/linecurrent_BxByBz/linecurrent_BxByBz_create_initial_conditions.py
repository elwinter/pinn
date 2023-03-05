#!/usr/bin/env python

"""Compute initial conditions for linecurrent_BxByBz.

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
description = "Compute initial conditions for linecurrent_BxByBz problem."

# Physical constants
μ0 = 1.0  # Normalized vacuum permeability

# Plasma parameters
m = 1.0    # Plasma article mass
ɣ = 5/3    # Adiabatic index = (N + 2)/N, N = # DOF=3, not 2.
n0 = 1.0   # Number density
P0 = 1.0   # Pressure
u0z = 0.0  # z-component of velocity
B0z = 0.0  # z-component of magnetic field
I = 1e-3   # Normalized current
C1 = μ0*I/(2*np.pi)  # Leading constant in analytical solutions for Bx, By.

# Define the constant fluid flow field.
θ = 60.0  # Angle in degrees clockwise from +y axis
u0 = 1.0  # Flow speed
u0x = u0*np.sin(np.radians(θ))  # x-component of flow velocity
u0y = u0*np.cos(np.radians(θ))  # y-component of flow velocity


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
        "-d", "--debug", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument('rest', nargs=argparse.REMAINDER)
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

    # Compute the initial conditions at spatial locations.
    # Each line is:
    # tg[0] x y Bx By Bz
    for x in xg:
        for y in yg:
            r = np.sqrt(x**2 + y**2)
            Bx = -C1*y/r**2
            By = C1*x/r**2
            Bz = B0z
            print(tg[0], x, y, Bx, By, Bz)


if __name__ == "__main__":
    """Begin main program."""
    main()
