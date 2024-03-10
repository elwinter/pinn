#!/usr/bin/env python

"""Create data for T problem.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
import problems.tsyganenko_ncs.T.T as p


# Program constants

# Program description.
description = 'Create data for T problem.'


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
        '--debug', '-d', action='store_true',
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
    if args.debug:
        print(f"args = {args}")
    debug = args.debug
    rest = args.rest

    # Fetch the remaining command-line arguments.
    # They should be in 3 sets of 3:
    # fPmin fPmax nfP fBzmin fBzmax nfBz phimin phimax nphi
    assert len(rest) == 9
    fPmin = float(rest[0])
    fPmax = float(rest[1])
    nfP = int(rest[2])
    fBzmin = float(rest[3])
    fBzmax = float(rest[4])
    nfBz = int(rest[5])
    phimin = float(rest[6])
    phimax = float(rest[7])
    nphi = int(rest[8])
    if debug:
        print(f"{fPmin} <= fP <= {fPmax}, nfP = {nfP}")
        print(f"{fBzmin} <= fBz <= {fBzmax}, nfBz = {nfBz}")
        print(f"{phimin} <= phi <= {phimax}, nphi = {nphi}")

    # Print the output header lines.
    header = '# GRID'
    print(header)
    header = (
        f"# {p.independent_variable_names[p.ifP]} "
        f"{p.independent_variable_names[p.ifBz]} "
        f"{p.independent_variable_names[p.iphi]}"
    )
    print(header)
    header = (
        f"# {fPmin} {fPmax} {nfP} "
        f"{fBzmin} {fBzmax} {nfBz} "
        f"{phimin} {phimax} {nphi}"
    )
    print(header)
    header = (
        f"# {p.independent_variable_names[p.ifP]} "
        f"{p.independent_variable_names[p.ifBz]} "
        f"{p.independent_variable_names[p.iphi]} "
        f"{p.dependent_variable_names[p.iT]}"
    )
    print(header)

    # Compute the data and send to stdout.
    fP = np.linspace(fPmin, fPmax, nfP)
    fBz = np.linspace(fBzmin, fBzmax, nfBz)
    phi = np.linspace(phimin, phimax, nphi)
    T = p.T_empirical(fP, fBz, phi)
    for _fP, _fBz, _phi, _T in zip(fP, fBz, phi, T):
        print(f"{_fP} {_fBz}, {_phi} {_T}")


if __name__ == '__main__':
    """Begin main program."""
    main()
