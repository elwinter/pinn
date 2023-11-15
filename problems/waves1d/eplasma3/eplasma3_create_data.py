#!/usr/bin/env python

"""Create data for eplasma3.

The data are the perturbation values for number density, x-velocity, and
x-electric field.

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
description = "Create data for eplasma3 problem."

me = 1.0    # Electron mass
e = 1.0     # Unit charge
eps0 = 1.0  # Permeability of free space

n0 = 1.0    # Ambient equilibrium number density
T = 1.0     # Ambient temperature

# Perturbation amplitudes for number density.
n1_amp = np.array([0.1, 0.1, 0.1])

# Wavelength and wavenumber of initial perturbations.
λ = np.array([0.5, 1.0, 2.0])
kx = 2*np.pi/λ

# Compute the electron plasma wave angular frequency for each component.
ω = plasma.electron_plasma_wave_angular_frequency(n0*me, T, kx, normalize=True)

# Amplitude of x-velocity perturbation
u1x_amp = ω/kx*n1_amp/n0

# Amplitude of Ex perturbation
E1x_amp = e*n1_amp/(kx*eps0)


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
    t = tg[0]
    for x in xg:
        phi = kx*x - ω*t
        n1 = np.sum(n1_amp*np.sin(phi))
        u1x = np.sum(u1x_amp*np.sin(phi))
        E1x = np.sum(E1x_amp*np.sin(phi + np.pi/2))
        print(t, x, n1, u1x, E1x)

    # Compute the boundary conditions at (t, x=0).
    # Each line is:
    # t xg[0] n1 u1x E1x
    x = xg[0]
    for t in tg:
        phi = kx*x - ω*t
        n1 = np.sum(n1_amp*np.sin(phi))
        u1x = np.sum(u1x_amp*np.sin(phi))
        E1x = np.sum(E1x_amp*np.sin(phi + np.pi/2))
        print(t, x, n1, u1x, E1x)


if __name__ == "__main__":
    """Begin main program."""
    main()
