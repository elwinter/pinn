#!/usr/bin/env python

"""Create data for beta problem.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
from pinn import common
import problems.tsyganenko_ncs.beta.beta as p


# Program constants

# Program description
DESCRIPTION = 'Create data for beta problem.'


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
    parser = common.create_minimal_command_line_argument_parser(DESCRIPTION)
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
    fBzmin = args.fBzmin
    fBzmax = args.fBzmax
    nfBz = args.nfBz

    # Print the output header lines.
    header = '# GRID'
    print(header)
    header = f"# {p.independent_variable_names[p.ifBz]}"
    print(header)
    header = f"# {fBzmin} {fBzmax} {nfBz}"
    print(header)
    header = (
        f"# {p.independent_variable_names[p.ifBz]} "
        f"{p.dependent_variable_names[p.ibeta]}"
    )
    print(header)

    # Compute the data and send to stdout.
    fBz = np.linspace(fBzmin, fBzmax, nfBz)
    beta = p.beta_empirical(fBz)
    for _fBz, _beta in zip(fBz, beta):
        print(f"{_fBz} {_beta}")


if __name__ == '__main__':
    """Begin main program."""
    main()
