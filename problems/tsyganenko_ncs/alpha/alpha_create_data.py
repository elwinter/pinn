#!/usr/bin/env python

"""Create data for the alpha problem.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
import pinn.common
import problems.tsyganenko_ncs.alpha.alpha as p


# Program constants

# Program description.
DESCRIPTION = 'Create data for the alpha problem.'


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
        'fPmin', type=float,
        help='Minimum value for fP'
    )
    parser.add_argument(
        'fPmax', type=float,
        help='Maximum value for fP'
    )
    parser.add_argument(
        'nfP', type=int,
        help='Number of fP steps'
    )
    parser.add_argument(
        'fBzmin', type=float,
        help='Minimum value for fBz'
    )
    parser.add_argument(
        'fBzmax', type=float,
        help='Maximum value for fBz'
    )
    parser.add_argument(
        'nfBz', type=int,
        help='Number of fBz steps'
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
    fPmin = args.fPmin
    fPmax = args.fPmax
    nfP = args.nfP
    fBzmin = args.fBzmin
    fBzmax = args.fBzmax
    nfBz = args.nfBz
    phimin = args.phimin
    phimax = args.phimax
    nphi = args.nphi

    # Print the output header lines.
    header = '# GRID'
    print(header)
    header = (
        f"# {p.independent_variable_names[p.ifP]}"
        f" {p.independent_variable_names[p.ifBz]}"
        f" {p.independent_variable_names[p.iphi]}"
    )
    print(header)
    header = (
        f"# {fPmin} {fPmax} {nfP}"
        f" {fBzmin} {fBzmax} {nfBz}"
        f" {phimin} {phimax} {nphi}"
    )
    print(header)
    header = (
        f"# {p.independent_variable_names[p.ifP]}"
        f" {p.independent_variable_names[p.ifBz]}"
        f" {p.independent_variable_names[p.iphi]}"
        f" {p.dependent_variable_names[p.ialpha]}"
    )
    print(header)

    # Compute the data and send to stdout.
    fP = np.linspace(fPmin, fPmax, nfP)
    fBz = np.linspace(fBzmin, fBzmax, nfBz)
    phi = np.linspace(phimin, phimax, nphi)
    for fP_ in fP:
        for fBz_ in fBz:
            for phi_ in phi:
                alpha = p.alpha_empirical(fP_, fBz_, phi_)
                print(f"{fP_} {fBz_} {phi_} {alpha}")


if __name__ == '__main__':
    """Begin main program."""
    main()
