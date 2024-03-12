#!/usr/bin/env python

"""Create data for the fP problem.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
import pinn.common
import problems.tsyganenko_ncs.fP.fP as p


# Program constants

# Program description.
DESCRIPTION = 'Create data for the fP problem.'


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
        'Pmin', type=float,
        help='Minimum value for P (nPa)'
    )
    parser.add_argument(
        'Pmax', type=float,
        help='Maximum value for P (nPa)'
    )
    parser.add_argument(
        'nP', type=int,
        help='Number of P steps'
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
    Pmin = args.Pmin
    Pmax = args.Pmax
    nP = args.nP

    # Print the output header lines.
    header = '# GRID'
    print(header)
    header = f"# {p.independent_variable_names[p.iP]}"
    print(header)
    header = f"# {Pmin} {Pmax} {nP}"
    print(header)
    header = (
        f"# {p.independent_variable_names[p.iP]}"
        f" {p.dependent_variable_names[p.ifP]}"
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
