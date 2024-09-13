#!/usr/bin/env python

"""Create data for the a1 problem.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
import pinn.common
import problems.tsyganenko_ncs.a1.a1 as p


# Program constants

# Program description
DESCRIPTION = 'Create data for the a1 problem.'


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
    parser = pinn.common.create_minimal_command_line_argument_parser(
        DESCRIPTION
    )
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

    # Print the output header lines.
    header = '# GRID'
    print(header)
    header = (
        f"# {p.independent_variable_names[p.ifP]}"
        f" {p.independent_variable_names[p.ifBz]}"
    )
    print(header)
    header = (
        f"# {fPmin} {fPmax} {nfP}"
        f" {fBzmin} {fBzmax} {nfBz}"
    )
    print(header)
    header = (
        f"# {p.independent_variable_names[p.ifP]}"
        f" {p.independent_variable_names[p.ifBz]}"
        f" {p.dependent_variable_names[p.ia1]}"
    )
    print(header)

    # Compute the data and send to stdout.
    fP = np.linspace(fPmin, fPmax, nfP)
    fBz = np.linspace(fBzmin, fBzmax, nfBz)
    for _fP in fP:
        for _fBz in fBz:
            _a1 = p.a1_empirical(_fP, _fBz)
            print(f"{_fP} {_fBz} {_a1}")


if __name__ == '__main__':
    """Begin main program."""
    main()
