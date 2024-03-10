#!/usr/bin/env python

"""Create plots for pinn0 results for fBz problem.

Create plots for pinn0 results for fBz problem.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

# Import standard modules.
from importlib import import_module
import os
import sys

# Import supplemental modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import project modules.
from pinn import common


# Program constants

# Program description
DESCRIPTION = 'Create plots for pinn0 results for fBz problem.'

# Name of directory to hold output plots
OUTPUT_DIR = 'pinn0_plots'

# Name of problem
PROBLEM_NAME = 'fBz'


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
    parser = common.create_minimal_command_line_argument_parser(DESCRIPTION)
    parser.add_argument(
        'results_path',
        help='Path to directory containing results to plot.'
    )
    parser.add_argument(
        'training_data_file',
        help='Name of file in results_path which contains training data. The file must include a PINN grid definition header.'
    )
    return parser


def main():
    """Main program."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}", flush=True)
    debug = args.debug
    verbose = args.verbose
    results_path = args.results_path
    training_data_file = args.training_data_file

    # Add the run results directory at the head of the module search path.
    sys.path.insert(0, results_path)

    # Import the problem definition from the run results directory.
    p = import_module(PROBLEM_NAME)

    # Compute the path to the output directory, then create it.
    output_path = OUTPUT_DIR
    os.mkdir(output_path)

    # Create the plots in a memory buffer.
    mpl.use("Agg")

    # -------------------------------------------------------------------------

    # Plot the loss history.
    if verbose:
        print(f"Plotting the loss history for {PROBLEM_NAME}.")

    # Load the loss data.
    path = os.path.join(results_path, 'L.dat')
    L = np.loadtxt(path)

    # Specify figure settings.
    figsize = (6.4, 4.8)  # This is the matplolib default.
    nrows, ncols = 1, 1

    # Create the figure.
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Loss function evolution for {p.dependent_variable_labels[p.ifBz]}"
    )
    gs = mpl.gridspec.GridSpec(nrows, ncols)

    # Create the plot.
    ax = fig.add_subplot(gs[0])
    ax.grid()
    ax.set_xlabel('Epoch')
    ax.set_xlim([0, L.size])
    ax.set_ylabel("$L$")
    ax.set_ylim([1e-3, 10.0])
    ax.grid(visible=True)

    # Plot the data, then add the legend.
    ax.semilogy(L, label="$L$")
    ax.legend()

    # Save the plot to a PNG file.
    path = os.path.join(output_path, 'L.png')
    fig.savefig(path)
    plt.close(fig)
    if verbose:
        print(f"Saved figure as {path}.")

    # ------------------------------------------------------------------------

    # Load the training points.
    path = os.path.join(results_path, training_data_file)
    training_data = np.loadtxt(path)

    # Read the data description from the header.
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline()  # GRID
        line = f.readline()  # Bz
        line = f.readline()  # Bzmin Bzmax nBz
        line = line[2:].rstrip()
        fields = line.split(' ')
        Bzmin = float(fields[0])
        Bzmax = float(fields[1])
        nBz = int(fields[2])

    # Find the epoch of the last trained model.
    last_model_epoch = common.find_last_epoch(results_path)

    # Load the trained model for each variable.
    models = []
    for variable_name in p.dependent_variable_names:
        path = os.path.join(results_path, 'models', f"{last_model_epoch}",
                            f"model_{variable_name}")
        model = tf.keras.models.load_model(path)
        models.append(model)

    # ------------------------------------------------------------------------

    # Plot the predicted and empirical solutions, and error, in a single
    # figure.
    variable_name = p.dependent_variable_names[p.ifBz]
    if verbose:
        print(f"Creating predicted/empirical/error figure for {variable_name}.")

    # Extract the training points, then compute the trained and empirical
    # solutions, and error.
    Bz_train = training_data[:, 0]
    fBz_trained = model(Bz_train).numpy().reshape(Bz_train.shape)
    fBz_empirical = p.fBz_empirical(Bz_train)
    fBz_error = fBz_trained - fBz_empirical

    # Compute the RMS error.
    fBz_rmserr = np.sqrt(np.sum(fBz_error**2)/fBz_error.size)

    # Specify figure settings.
    figsize = (6.4, 4.8)  # This is the matplolib default.
    nrows, ncols = 1, 1

    # Create the figure.
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        "Comparison of trained and empirical solutions for $f_{Bz}$"
    )
    gs = mpl.gridspec.GridSpec(nrows, ncols)

    # Create the plot.
    ax = fig.add_subplot(gs[0])
    ax.grid()
    ax.set_xlabel("$B_z$")
    ax.set_xlim([Bzmin, Bzmax])
    ax.set_ylabel("$f_{Bz}$")
    fBzmin, fBzmax = -2.0, 2.0
    ax.set_ylim([fBzmin, fBzmax])
    ax.grid(visible=True)

    # Plot the data, then add the legend.
    ax.plot(Bz_train, fBz_trained, label="$f_{Bz}$ (trained)")
    ax.plot(Bz_train, fBz_empirical, label="$f_{Bz}$ (empirical)")
    ax.plot(Bz_train, fBz_error, label="$f_{Bz}$ error")
    ax.legend()

    # Add a plot title with the RMS error.
    text = f"RMS error = {fBz_rmserr:.2E}"
    ax.set_title(text)

    # Save the plot to a PNG file.
    path = os.path.join(output_path, 'fBz.png')
    fig.savefig(path)
    plt.close(fig)
    if verbose:
        print(f"Saved figure as {path}.")


if __name__ == '__main__':
    """Begin main program."""
    main()
