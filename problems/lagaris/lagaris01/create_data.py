#!/usr/bin/env python

"""Create data for lagaris01 problem.

This is for problem 1 from Lagaris (1998).

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse
import copy
import sys

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.
from pinn import common
from problems.lagaris.lagaris01 import lagaris01 as p


# Program constants

# Program description.
DESCRIPTION = "Create data for lagaris01 problem."

# Default values for command-line arguments when none are supplied (such as
# when create_data() is called by external code).
args_default = {
    "debug": False,
    "verbose": False,
}


def create_command_line_parser(description: str = DESCRIPTION):
    """Create the command-line parser.

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
    parser = common.create_minimal_command_line_parser(DESCRIPTION)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    return parser


def create_data(args: dict):
    """Create data for the lagaris01 problem.

    Create data for the lagaris01 problem. Results are sent to standard
    output.

    Parameters
    ----------
    args : dict
        Dictionary of command-line and other options.

    Returns
    -------
    0 on success.

    Raises
    ------
    None
    """
    # Use defaults for unspecified arguments. Merge additional settings
    # which are passed in by the caller.
    local_args = copy.deepcopy(args_default)
    if args is not None:
        local_args.update(args)
    args = local_args

    # Local convenience variables.
    debug = args["debug"]
    verbose = args["verbose"]
    rest = args["rest"]

    # ------------------------------------------------------------------------

    # Fetch the remaining command-line arguments.
    # They should be in a set of 3:
    # xmin xmax nx
    xmin = float(rest[0])
    xmax = float(rest[1])
    nx = int(rest[2])
    if debug: print(f"{xmin} <= x <= {xmax}, nx = {nx}")

    # Print the output header lines.
    header = f"# GRID x {xmin} {xmax} {nx}"
    print(header)
    header = "# x Ψ"
    print(header)

    # Compute the value at each grid point.
    # Each line is:
    # x Ψ
    x = np.linspace(xmin, xmax, nx)
    for _x in x:
        Ψ = p.Ψ_analytical(_x)
        print(_x, Ψ)

    # ------------------------------------------------------------------------

    # Return normally.
    return 0


def main():
    """Top-level code for the command-line version of create_data.

    This is the top-level code for the command-line version of create_data.
    It processes command-line options, then calls the create_data() function.

    Parameters
    ----------
    None

    Returns
    -------
    0 on success

    Raises
    ------
    None
    """
    # Set up the command-line parser.
    parser = create_command_line_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}")

    # Convert the arguments from Namespace to dict.
    args = vars(args)

    # Call the main program logic.
    return_code = create_data(args)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
