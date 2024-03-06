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
description = "Create data for T problem."


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
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER)
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
    # fPmin fP_max nfP
    assert len(rest) == 3
    fPmn = float(rest[0])
    fPmax = float(rest[1])
    nfP = int(rest[2])
    if debug:
        print(f"{fPmin} <= fP <= {fPmax}, nfP = {nfP}")

    # Print the output header lines.
    header = "# GRID"
    print(header)
    header = "# T"
    print(header)
    header = f"# {fPmin} {fPmax} {nfP}"
    print(header)
    header = "# fP T"
    print(header)

    # Compute the data..
    # Each line is:
    # fP T
    fP = np.linspace(fPmin, fPmax, nfP)
    T = p.T_analytical(fP)
    for fp, t in zip(fP, T):
        print(f"{fP} {T}")



if __name__ == "__main__":
    """Begin main program."""
    main()
