#!/usr/bin/env python

"""Create data for the wave_equation_1d problem.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
import pinn.common
import problems.simple.wave_equation_1d.wave_equation_1d as p


# Program constants

# Program description
DESCRIPTION = 'Create data for the wave_equation_1d problem.'


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
        'tmin', type=float,
        help='Minimum value for t'
    )
    parser.add_argument(
        'tmax', type=float,
        help='Maximum value for t'
    )
    parser.add_argument(
        'nt', type=int,
        help='Number of t steps'
    )
    parser.add_argument(
        'xmin', type=float,
        help='Minimum value for x'
    )
    parser.add_argument(
        'xmax', type=float,
        help='Maximum value for x'
    )
    parser.add_argument(
        'nx', type=int,
        help='Number of x steps'
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
    tmin = args.tmin
    tmax = args.tmax
    nt = args.nt
    xmin = args.xmin
    xmax = args.xmax
    nx = args.nx

    # Print the output header lines.
    header = '# GRID'
    print(header)
    header = f"# {' '.join(p.independent_variable_names)}"
    print(header)
    header = (
        f"# {tmin} {tmax} {nt}"
        f" {xmin} {xmax} {nx}"
    )
    print(header)
    header = f"# {' '.join(p.independent_variable_names + p.dependent_variable_names)} "
    print(header)

    # Compute the data and send to stdout.
    t = np.linspace(tmin, tmax, nt)
    x = np.linspace(xmin, xmax, nx)
    for _t in t:
        for _x in x:
            _y = p.y_analytical(_t, _x)
            print(f"{_t} {_x} {_y}")


if __name__ == '__main__':
    """Begin main program."""
    main()
