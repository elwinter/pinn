#!/usr/bin/env python

"""Create data for fBz problem.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
from pinn import common
import problems.tsyganenko_ncs.fBz.fBz as p


# Program constants

# Program description
DESCRIPTION = 'Create data for fBz problem.'


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
        'Bzmin', type=float,
        help='Minimum value for Bz (nT)'
    )
    parser.add_argument(
        'Bzmax', type=float,
        help='Maximum value for Bz (nT)'
    )
    parser.add_argument(
        'nBz', type=int,
        help='Number of Bz steps'
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
    Bzmin = args.Bzmin
    Bzmax = args.Bzmax
    nBz = args.nBz

    # Print the output header lines.
    header = '# GRID'
    print(header)
    header = f"# {p.independent_variable_names[p.iBz]}"
    print(header)
    header = f"# {Bzmin} {Bzmax} {nBz}"
    print(header)
    header = (
        f"# {p.independent_variable_names[p.iBz]} "
        f"{p.dependent_variable_names[p.ifBz]}"
    )
    print(header)

    # Compute the data and send to stdout.
    Bz = np.linspace(Bzmin, Bzmax, nBz)
    fBz = p.fBz_empirical(Bz)
    for _Bz, _fBz in zip(Bz, fBz):
        print(f"{_Bz} {_fBz}")


if __name__ == '__main__':
    """Begin main program."""
    main()
