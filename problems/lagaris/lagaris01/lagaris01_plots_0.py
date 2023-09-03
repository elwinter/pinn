#!/usr/bin/env python

"""Create plots for pinn0 results for lagaris01 problem.

Create plots for pinn0 results for lagaris01 problem.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

# Import standard modules.
import argparse
import os

# Import supplemental modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Import project modules.
import pinn.standard_plots as psp


# Program constants

# Program description.
DESCRIPTION = "Create plots for pinn0 results for lagaris01 problem."

# Name of directory to hold output plots.
OUTPUT_DIR = "plots_0"


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
    """
    parser = argparse.ArgumentParser(DESCRIPTION)
    parser.add_argument(
        "--clobber", action="store_true",
        help="Clobber existing output directory (default: %(default)s)."
    )
    parser.add_argument(
        "-d", "--debug", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "results_path",
        help="Path to directory containing results to plot."
    )
    return parser


def main():
    """Main program."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    clobber = args.clobber
    debug = args.debug
    verbose = args.verbose
    results_path = args.results_path
    if debug:
        print(f"args = {args}", flush=True)

    # Compute the path to the output directory. Then create it if needed.
    output_path = OUTPUT_DIR
    if os.path.exists(output_path):
        if not clobber:
            raise FileExistsError(f"Output directory {output_path} exists!")
    else:
        os.mkdir(output_path)

    # -------------------------------------------------------------------------

    # Plot the data loss history.

    # Read the data.
    path = os.path.join(results_path, "L_data.dat")
    L_dat = np.loadtxt(path)
    if debug:
        print(f"L_dat = {L_dat}")

    # Create the plot in a memory buffer.
    mpl.use("Agg")

    # Create the plot. 
    plt.semilogy(L_dat)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Data loss")
    plt.grid()

    # Save the plot to a PNG file.
    path = os.path.join(output_path, "L_dat.png")
    plt.savefig(path)

    # Return to standard plotting backend.
    mpl.use("TkAgg")


if __name__ == "__main__":
    """Begin main program."""
    main()
