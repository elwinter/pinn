#!/usr/bin/env python

"""Create data for fBz problem.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
import problems.tsyganenko_ncs.fBz.fBz as p


# Program constants

# Program description.
description = 'Create data for fBz problem.'


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
    # They should be in a set of 3:
    # Bzmin Bzmax nBz
    assert len(rest) == 3
    Bzmin = float(rest[0])
    Bzmax = float(rest[1])
    nBz = int(rest[2])
    if debug:
        print(f"{Bzmin} <= Bz <= {Bzmax}, nx = {nBz}")

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
