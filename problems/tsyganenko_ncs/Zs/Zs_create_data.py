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
        'rhomin', type=float,
        help='Minimum value for rho'
    )
    parser.add_argument(
        'rhomax', type=float,
        help='Maximum value for rho'
    )
    parser.add_argument(
        'nrho', type=int,
        help='Number of rho steps'
    )
    parser.add_argument(
        'phimin', type=float,
        help='Minimum value for phi'
    )
    parser.add_argument(
        'phimax', type=float,
        help='Maximum value for phi'
    )
    parser.add_argument(
        'nphi', type=int,
        help='Number of phi steps'
    )
    parser.add_argument(
        'Pmin', type=float,
        help='Minimum value for P'
    )
    parser.add_argument(
        'Pmax', type=float,
        help='Maximum value for P'
    )
    parser.add_argument(
        'nP', type=int,
        help='Number of P steps'
    )
    parser.add_argument(
        'Bymin', type=float,
        help='Minimum value for By'
    )
    parser.add_argument(
        'Bymax', type=float,
        help='Maximum value for By'
    )
    parser.add_argument(
        'nBy', type=int,
        help='Number of By steps'
    )
    parser.add_argument(
        'Bzmin', type=float,
        help='Minimum value for Bz'
    )
    parser.add_argument(
        'Bzmax', type=float,
        help='Maximum value for Bz'
    )
    parser.add_argument(
        'nBz', type=int,
        help='Number of Bz steps'
    )
    parser.add_argument(
        'psimin', type=float,
        help='Minimum value for psi'
    )
    parser.add_argument(
        'psimax', type=float,
        help='Maximum value for psi'
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
    rhomin = args.rhomin
    rhomax = args.rhomax
    nrho = args.nrho
    phimin = args.phimin
    phimax = args.phimax
    nphi = args.nphi
    Pmin = args.Pmin
    Pmax = args.Pmax
    nP = args.nP
    Bymin = args.Bymin
    Bymax = args.Bymax
    nBy = args.nBy
    Bzmin = args.Bzmin
    Bzmax = args.Bzmax
    nBz = args.nBz
    psimin = args.phimin
    psimax = args.psimax
    npsi = args.npsi

    # Print the output header lines.
    header = '# GRID'
    print(header)
    header = (
        f"# {p.independent_variable_names[p.irho]}"
        f" {p.independent_variable_names[p.iphi]}"
        f" {p.independent_variable_names[p.iP]}"
        f" {p.independent_variable_names[p.iBy]}"
        f" {p.independent_variable_names[p.iBz]}"
        f" {p.independent_variable_names[p.ipsi]}"
    )
    print(header)
    header = (
        f"# {rhomin} {rhomax} {nrho}"
        f" {phimin} {phimax} {nphi}"
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

    # Compute the data and send to stdout.
    rho = np.linspace(rhomin, rhomax, nrho)
    phi = np.linspace(phimin, phimax, nphi)
    P = np.linspace(Pmin, Pmax, nP)
    By = np.linspace(Bymin, Bymax, nBy)
    Bz = np.linspace(Bzmin, Bzmax, nBz)
    psi = np.linspace(psimin, psimax, npsi)
    for _rho in rho:
        for _phi in phi:
            for _P in P:
                for _By in By:
                    for _Bz in Bz:
                        for _psi in psi:
                            _Zs = p.Zs_empirical(_rho, _phi,
                                                 _P, _By, _Bz, _psi)
                print(f"{_rho} {_phi} {_P} {_By} {_Bz} {_psi} {_Zs}")


if __name__ == '__main__':
    """Begin main program."""
    main()
