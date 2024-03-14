#!/usr/bin/env python

"""Create data for the Zs problem.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
import pinn.common
import problems.tsyganenko_ncs.Zs.Zs as p


# Program constants

# Program description.
DESCRIPTION = 'Create data for the Zs problem.'


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
    parser = pinn.common.create_minimal_command_line_argument_parser(DESCRIPTION)
    parser.add_argument(
        '--cartesian', action='store_true',
        help='Use Cartesian instead of cylindrical coordinates (treat rho as'
             ' x, treat phi as y, on the command line). This creates a a grid'
             ' in the XY plane, but presents the points in that plane in'
             ' (rho, phi) coordinates, as expected by the Zs() model.'
    )
    parser.add_argument(
        'rhoxmin', type=float,
        help='Minimum value for rho (R_E) or x (R_E) (SM frame)'
    )
    parser.add_argument(
        'rhoxmax', type=float,
        help='Maximum value for rho (R_E) or x (R_E) (SM frame)'
    )
    parser.add_argument(
        'nrhox', type=int,
        help='Number of rho (or x) steps'
    )
    parser.add_argument(
        'phiymin', type=float,
        help='Minimum value for phi (degrees > 0) or y (R_E) (SM frame)'
    )
    parser.add_argument(
        'phiymax', type=float,
        help='Maximum value for phi (degrees > 0) or y (R_E) (SM frame)'
    )
    parser.add_argument(
        'nphiy', type=int,
        help='Number of phi (or y) steps'
    )
    parser.add_argument(
        'Pmin', type=float,
        help='Minimum value for P (nPa)'
    )
    parser.add_argument(
        'Pmax', type=float,
        help='Maximum value for P (nPa)'
    )
    parser.add_argument(
        'nP', type=int,
        help='Number of P steps'
    )
    parser.add_argument(
        'Bymin', type=float,
        help='Minimum value for By (nT, SM frame)'
    )
    parser.add_argument(
        'Bymax', type=float,
        help='Maximum value for By (nT, SM frame)'
    )
    parser.add_argument(
        'nBy', type=int,
        help='Number of By steps'
    )
    parser.add_argument(
        'Bzmin', type=float,
        help='Minimum value for Bz (nT, SM frame)'
    )
    parser.add_argument(
        'Bzmax', type=float,
        help='Maximum value for Bz (nT, SM frame)'
    )
    parser.add_argument(
        'nBz', type=int,
        help='Number of Bz steps'
    )
    parser.add_argument(
        'psimin', type=float,
        help='Minimum value for psi (degrees, SM frame)'
    )
    parser.add_argument(
        'psimax', type=float,
        help='Maximum value for psi (degrees, SM frame)'
    )
    parser.add_argument(
        'npsi', type=int,
        help='Number of psi steps'
    )
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
    verbose = args.verbose
    cartesian = args.cartesian
    rhoxmin = args.rhoxmin
    rhoxmax = args.rhoxmax
    nrhox = args.nrhox
    phiymin = args.phiymin
    phiymax = args.phiymax
    nphiy = args.nphiy
    Pmin = args.Pmin
    Pmax = args.Pmax
    nP = args.nP
    Bymin = args.Bymin
    Bymax = args.Bymax
    nBy = args.nBy
    Bzmin = args.Bzmin
    Bzmax = args.Bzmax
    nBz = args.nBz
    psimin = args.psimin
    psimax = args.psimax
    npsi = args.npsi

    # Convert angles to radians.
    if not cartesian:
        phiymin = np.radians(phiymin)
        phiymax = np.radians(phiymax)
    psimin = np.radians(psimin)
    psimax = np.radians(psimax)

    # Print the output header lines.
    header = '# GRID'
    print(header)
    if cartesian:
        header = '# x y'
    else:
        header = (
            f"# {p.independent_variable_names[p.irho]}"
            f" {p.independent_variable_names[p.iphi]}"
        )
    header += (
        f" {p.independent_variable_names[p.iP]}"
        f" {p.independent_variable_names[p.iBy]}"
        f" {p.independent_variable_names[p.iBz]}"
        f" {p.independent_variable_names[p.ipsi]}"
    )
    print(header)
    header = (
        f"# {rhoxmin} {rhoxmax} {nrhox}"
        f" {phiymin} {phiymax} {nphiy}"
        f" {Pmin} {Pmax} {nP}"
        f" {Bymin} {Bymax} {nBy}"
        f" {Bzmin} {Bzmax} {nBz}"
        f" {psimin} {psimax} {npsi}"
    )
    print(header)
    header = (
        f"# {p.independent_variable_names[p.irho]}"
        f" {p.independent_variable_names[p.iphi]}"
        f" {p.independent_variable_names[p.iP]}"
        f" {p.independent_variable_names[p.iBy]}"
        f" {p.independent_variable_names[p.iBz]}"
        f" {p.independent_variable_names[p.ipsi]}"
        f" {p.dependent_variable_names[p.iZs]}"
    )
    print(header)

    # Create the grid in Cartesian or cylindrical coordinates.
    rhox = np.linspace(rhoxmin, rhoxmax, nrhox)
    phiy = np.linspace(phiymin, phiymax, nphiy)
    P = np.linspace(Pmin, Pmax, nP)
    By = np.linspace(Bymin, Bymax, nBy)
    Bz = np.linspace(Bzmin, Bzmax, nBz)
    psi = np.linspace(psimin, psimax, npsi)

    # Compute the data and send to stdout.
    # Iterate across the Cartesian or cylindrical grid.
    for _rhox in rhox:
        for _phiy in phiy:
            if cartesian:
                _rho = np.sqrt(_rhox**2 + _phiy**2)
                _phi = np.arctan2(_phiy, _rhox)
                if _phi < 0:
                    _phi += 2*np.pi
            else:
                _rho = _rhox
                _phi = _phiy
            for _P in P:
                for _By in By:
                    for _Bz in Bz:
                        for _psi in psi:
                            _Zs = p.Zs_empirical(
                                _rho, _phi, _P, _By, _Bz, _psi
                            )
                            print(
                                f"{_rho} {_phi} {_P} {_By} {_Bz} {_psi} {_Zs}"
                            )

if __name__ == '__main__':
    """Begin main program."""
    main()
