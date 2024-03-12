#!/usr/bin/env python

"""Create plots for pinn0 results for the fBz problem.

Create plots for pinn0 results for the fBz problem.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

# Import standard modules.
import os

# Import supplemental modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import project modules.
import pinn.common


# Program constants

# Program description
DESCRIPTION = 'Create plots for pinn0 results for the fBz problem.'

# Name of directory to hold output plots
OUTPUT_DIR = 'pinn0_plots'

# Name and file of problem (in output directory)
PROBLEM_NAME = 'fBz'
PROBLEM_FILE = f"{PROBLEM_NAME}.py"


def create_command_line_argument_parser():
    """Create the command-line argument parser.

    Create the command-line argument parser.

    Parameters
    ----------
    None

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser for command-line arguments

    Raises
    ------
    None
    """
    parser = pinn.common.create_minimal_command_line_argument_parser(
        DESCRIPTION
    )
    parser.add_argument(
        'results_directory',
        help='Path to directory containing results to plot'
    )
    parser.add_argument(
        'training_data_file',
        help='Name of file in results_directory which contains training data'
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
    results_directory = args.results_directory
    training_data_file = args.training_data_file

    # -------------------------------------------------------------------------

    # Import the problem definition from the run results directory.
    path = os.path.join(results_directory, PROBLEM_FILE)
    p = pinn.common.import_problem(path)

    # Compute the path to the plot output directory, then create it.
    output_path = OUTPUT_DIR
    os.mkdir(output_path)

    # Create the plots in a memory buffer.
    mpl.use('Agg')

    # -------------------------------------------------------------------------

    # Plot the loss history.
    if verbose:
        print(f"Plotting the loss history for {PROBLEM_NAME}.")

    # Load the loss data.
    path = os.path.join(results_directory, 'L.dat')
    L = np.loadtxt(path)

    # Specify figure settings.
    figsize = (6.4, 4.8)  # This is the matplolib default.
    nrows, ncols = 1, 1
    ivar = p.ifBz
    varname = p.dependent_variable_names[ivar]
    varlabel = p.dependent_variable_labels[ivar]
    suptitle = f"Loss function evolution for {varlabel}"
    xlabel = 'Epoch'
    xlim = [0, L.size]
    ylabel = '$L$'
    ylim = [1e-3, 10.0]
    plot_filename = 'L.png'

    # Create the figure.
    fig = plt.figure(figsize=figsize)
    fig.suptitle(suptitle)
    gs = mpl.gridspec.GridSpec(nrows, ncols)

    # Create the plot.
    ax = fig.add_subplot(gs[0])
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.grid()

    # Plot the data, then add the legend.
    ax.semilogy(L, label=ylabel)
    ax.legend()

    # Save the plot to a PNG file.
    path = os.path.join(output_path, plot_filename)
    fig.savefig(path)
    plt.close(fig)
    if verbose:
        print(f"Saved figure as {path}.")

    # ------------------------------------------------------------------------

    # Load the training points, and count them.
    path = os.path.join(results_directory, training_data_file)
    column_descriptions, training_data = pinn.common.read_grid_file(path)
    ivname, ivmin, ivmax, ivn = 0, 1, 2, 3  # Description field indices
    n_train = training_data.shape[0]

    # Extract the training data needed for this plot.

    # Find the epoch of the last trained model.
    last_model_epoch = pinn.common.find_last_epoch(results_directory)

    # Load the trained model for each variable.
    models = []
    for varname in p.dependent_variable_names:
        path = os.path.join(results_directory, 'models', f"{last_model_epoch}",
                            f"model_{varname}")
        model = tf.keras.models.load_model(path)
        models.append(model)

    # ------------------------------------------------------------------------

    # Plot the predicted and empirical solutions, and error, in a single
    # figure.

    # Get indices and function from the problem definition.
    ix = p.iBz
    iy = p.ifBz
    empirical = p.fBz_empirical

    # Extract training data needed for this plot.
    x_train = training_data[:, ix]  # Shape (n_train,)
    y_train = training_data[:, p.n_dim + iy]  # Shape (n_train,)

    # Compute the trained and empirical solutions, error, and RMSE.
    y_trained = models[iy](x_train).numpy().reshape((n_train,))
    y_empirical = empirical(x_train)
    y_error = y_trained - y_empirical
    y_rmserr = np.sqrt(np.sum(y_error**2)/y_error.size)

    # Compute plot settings.
    varname = p.dependent_variable_names[iy]
    varlabel = p.dependent_variable_labels[iy]
    xlabel = p.independent_variable_labels[ix]
    xlim = [column_descriptions[ix][ivmin], column_descriptions[ix][ivmax]]
    ylabel = varlabel
    ylim = [-2.0, 2.0]  # <HACK/>
    training_point_color = 'black'
    figsize = (6.4, 4.8)  # This is the matplolib default.
    nrows, ncols = 1, 1
    plot_filename = f"{varname}.png"

    if verbose:
        print(f"Creating predicted/empirical/error figure for {varname}.")

    # Create the figure.
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Comparison of trained and empirical solutions for {varlabel}"
    )
    gs = mpl.gridspec.GridSpec(nrows, ncols)

    # Create the plot.
    ax = fig.add_subplot(gs[0])
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.grid()

    # Plot the data, then add the legend.
    # Use scatter() to show training point locations.
    ax.scatter(x_train, y_train, c=training_point_color,
               label=f"{varlabel} (training)")
    ax.plot(x_train, y_trained, label=f"{varlabel} (trained)")
    ax.plot(x_train, y_empirical, label=f"{varlabel} (empirical)")
    ax.plot(x_train, y_error, label=f"{varlabel} error")
    ax.legend()

    # Add a plot title with the RMS error.
    text = f"RMS error = {y_rmserr:.2E}"
    ax.set_title(text)

    # Save the plot to a PNG file.
    path = os.path.join(output_path, plot_filename)
    fig.savefig(path)
    plt.close(fig)
    if verbose:
        print(f"Saved figure as {path}.")


if __name__ == '__main__':
    """Begin main program."""
    main()
