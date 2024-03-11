#!/usr/bin/env python

"""Create data for T problem.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
from pinn import common
import problems.tsyganenko_ncs.T.T as p


# Program constants

# Program description
DESCRIPTION = 'Create data for T problem.'


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

    # Print the output header lines.
    header = '# GRID'
    print(header)
    header = f"# {p.independent_variable_names[p.ifP]}"
    print(header)
    header = f"# {fPmin} {fPmax} {nfP}"
    print(header)
    header = (
        f"# {p.independent_variable_names[p.ifP]} "
        f"{p.dependent_variable_names[p.iT]}"
    )
    print(header)

    # Compute the data and send to stdout.
    fP = np.linspace(fPmin, fPmax, nfP)
    T = p.T_empirical(fP)
    for _fP, _T in zip(fP, T):
        print(f"{_fP} {_T}")


if __name__ == '__main__':
    """Begin main program."""
    main()
