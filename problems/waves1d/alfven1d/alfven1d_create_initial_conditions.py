#!/usr/bin/env python

"""Compute initial conditions for alfven1d.

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


# Program constants

# Program description.
description = "Compute initial conditions for alfven1d problem."

me = 1.0    # Electron mass
n0 = 1.0    # Ambient equilibrium number density
T = 1.0     # Ambient temperature

# Perturbation amplitudes for dependent variables (dimensionless).
u1y_amp = 0.1
B1y_amp = 0.1

# Wavelength and wavenumber of initial perturbations.
λ = 1.0
kx = 2*np.pi/λ

# Compute the electron plasma wave angular frequency for each component.
ω = plasma.electron_plasma_wave_angular_frequency(n0*me, T, kx, normalize=True)


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
    # They should be in 2 sets of 3:
    # t_min t_max n_t x_min x_max n_x
    assert len(rest) == 6
    (t_min, x_min,) = np.array(rest[::3], dtype=float)
    (t_max, x_max,) = np.array(rest[1::3], dtype=float)
    (n_t, n_x,) = np.array(rest[2::3], dtype=int)
    if debug:
        print(f"{t_min} <= t <= {t_max}, n_t = {n_t}")
        print(f"{x_min} <= x <= {x_max}, n_x = {n_x}")

    # Create the grid points for the boundary conditions.
    tg = np.linspace(t_min, t_max, n_t)
    if debug:
        print(f"tg = {tg}")

    # Create the grid points for the initial conditions.
    xg = np.linspace(x_min, x_max, n_x)
    if debug:
        print(f"xg = {xg}")

    # Compute the boundary conditions at time intervals.
    # Each line is:
    # t x_min n1 P1 u1x u1y u1z B1x B1y B1z
    x = 0.0
    for t in tg:
        n1 = 0.0
        P1 = 0.0
        u1x = 0.0
        u1y = u1y_amp*np.sin(kx*x - ω*t)
        u1z = 0.0
        B1x = 0.0
        B1y = B1y_amp*np.sin(kx*x - ω*t + np.pi)
        B1z = 0.0
        print(t, x, n1, P1, u1x, u1y, u1z, B1x, B1y, B1z)

    # Compute the initial conditions at spatial locations.
    # Each line is:
    # t_min x n1 P1 u1x u1y u1z B1x B1y B1z
    t = 0.0
    for x in xg:
        n1 = 0.0
        P1 = 0.0
        u1x = 0.0
        u1y = u1y_amp*np.sin(kx*x - ω*t)
        u1z = 0.0
        B1x = 0.0
        B1y = B1y_amp*np.sin(kx*x - ω*t + np.pi)
        B1z = 0.0
        print(t, x, n1, P1, u1x, u1y, u1z, B1x, B1y, B1z)


if __name__ == "__main__":
    """Begin main program."""
    main()
