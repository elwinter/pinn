#!/usr/bin/env python


"""Plot the residual, data, and total loss functions by model.

Plot the residual, data, and total loss functions by model.
The plots are saved as a single PNG file.
"""


# Import standard modules.
import argparse
import importlib
import os
import sys

# # Import 3rd-party modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Import project-specific modules.
from pinn import standard_plots


# Program constants and defaults

# Program description.
DESCRIPTION = (
    "Plot the residual, data, and total loss functions by model."
)

# Default maximum and minimum values to show in loss function plots.
DEFAULT_LMIN = 1.0e-9
DEFAULT_LMAX = 1.0


def create_command_line_parser():
    """Create the command-line argument parser.

    Create the command-line argument parser.

    Parameters
    ----------
    None

    Returns
    -------
    parser : argparse.ArgumentParser
        Command-line argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--Lmax", type=float, default=DEFAULT_LMAX,
        help="Maximum L value to plot (default: %(default)s)."
    )
    parser.add_argument(
        "--Lmin", type=float, default=DEFAULT_LMIN,
        help="Minimum L value to plot (default: %(default)s)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "problem_name", type=str,
        help="Name of problem."
    )
    parser.add_argument(
        "results_path", type=str, nargs="?", default=os.getcwd(),
        help="Directory containing model results (default: %(default)s)."
    )
    return parser


def main():
    """Main program logic.

    The program starts here.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Set up the command-line parser.
    parser = create_command_line_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    debug = args.debug
    Lmax = args.Lmax
    Lmin = args.Lmin
    verbose = args.verbose
    problem_name = args.problem_name
    results_path = args.results_path
    if debug:
        print(f"args = {args}")
        print(f"debug = {debug}")
        print(f"Lmax = {Lmax}")
        print(f"Lmin = {Lmin}")
        print(f"verbose = {verbose}")
        print(f"problem_name = {problem_name}")
        print(f"results_path = {results_path}")

    # Import the problem definition.
    sys.path.append(results_path)
    p = importlib.import_module(problem_name)

    # Load the overall, residual, and data losses for the each model.
    losses_model = np.loadtxt(os.path.join(results_path, "losses_model.dat"))
    losses_model_res = np.loadtxt(os.path.join(results_path, "losses_model_res.dat"))
    losses_model_data = np.loadtxt(os.path.join(results_path, "losses_model_data.dat"))

    # Plot the loss functions and save as a PNG.
    mpl.use("Agg")
    standard_plots.plot_model_loss_functions(
        losses_model_res, losses_model_data, losses_model,
        p.dependent_variable_labels
    )
    plt.savefig("L_models.png")


if __name__ == "__main__":
    """Begin main program."""
    main()
