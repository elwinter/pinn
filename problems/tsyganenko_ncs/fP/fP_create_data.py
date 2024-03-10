#!/usr/bin/env python

"""Create data for fP problem.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
import problems.tsyganenko_ncs.fP.fP as p


# Program constants

# Program description.
description = "Create data for fP problem."


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
    # Pmin Pmax nP
    assert len(rest) == 3
    Pmin = float(rest[0])
    Pmax = float(rest[1])
    nP = int(rest[2])
    if debug:
        print(f"{Pmin} <= P <= {Pmax}, nx = {nP}")

    # Print the output header lines.
    header = '# GRID'
    print(header)
    header = f"# {p.independent_variable_names[p.iP]}"
    print(header)
    header = f"# {Pmin} {Pmax} {nP}"
    print(header)
    header = (
        f"# {p.independent_variable_names[p.iP]} "
        f"{p.dependent_variable_names[p.ifP]}"
    )
    print(header)

    # Compute the data and send to stdout.
    P = np.linspace(Pmin, Pmax, nP)
    fP = p.fP_empirical(P)
    for _P, _fP in zip(P, fP):
        print(f"{_P} {_fP}")


if __name__ == '__main__':
    """Begin main program."""
    main()
