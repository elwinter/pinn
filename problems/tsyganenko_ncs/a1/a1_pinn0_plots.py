#!/usr/bin/env python

"""Create plots for pinn0 results for the a1 problem.

Create plots for pinn0 results for the a1 problem.

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
DESCRIPTION = 'Create plots for pinn0 results for the a1 problem.'

# Name of directory to hold output plots
OUTPUT_DIR = 'pinn0_plots'

# Name of problem
PROBLEM_NAME = 'a1'
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
        Parser for command-line arguments.
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
        help='Name of file in results_directory which contains training data. The file must include a PINN grid definition header.'
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

    # ------------------------------------------------------------------------

    # Load the training points, and count them.
    path = os.path.join(results_directory, training_data_file)
    column_descriptions, training_data = pinn.common.read_grid_file(path)
    ivname, ivmin, ivmax, ivn = 0, 1, 2, 3  # Description field indices
    n_train = training_data.shape[0]

    # Find the epoch of the last trained model.
    last_model_epoch = pinn.common.find_last_epoch(results_directory)

    # Load the trained model for each variable.
    models = []
    for varname in p.dependent_variable_names:
        path = os.path.join(results_directory, 'models', f"{last_model_epoch}",
                            f"model_{varname}")
        model = tf.keras.models.load_model(path)
        models.append(model)

    # -------------------------------------------------------------------------

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
    ivar = p.ia1
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

    # # Get indices and function from the problem definition.
    ix = p.ifP
    iy = p.ifBz
    iz = p.ia1
    empirical = p.a1_empirical

    # Extract training data needed for this plot.
    x_train = training_data[:, ix]  # Shape (n_train,)
    y_train = training_data[:, iy]  # Shape (n_train,)
    xy_train = training_data[:, ix:iy + 1]  # Shape (n_train, 2)
    z_train = training_data[:, p.n_dim + iz]  # Shape (n_train,)
    nx = column_descriptions[ix][ivn]
    ny = column_descriptions[iy][ivn]

    # Compute the trained and empirical solutions, error, and RMSE.
    z_trained = models[iz](xy_train).numpy().reshape((n_train,))
    z_empirical = empirical(x_train, y_train)
    z_error = z_trained - z_empirical
    z_rmserr = np.sqrt(np.sum(z_error**2)/z_error.size)

    # Reshape the coordinates and values into 2D arrays.
    # These should look as if they were created with meshgrid() from
    # linspace() arrays of x and y.
    X = x_train.reshape(nx, ny)
    Y = y_train.reshape(nx, ny)
    Z_train = z_train.reshape(nx, ny)
    Z_trained = z_trained.reshape(nx, ny)
    Z_empirical = z_empirical.reshape(nx, ny)
    Z_error = z_error.reshape(nx, ny)

    # Compute plot settings.
    varname = p.dependent_variable_names[iz]
    varlabel = p.dependent_variable_labels[iz]
    xlabel = p.independent_variable_labels[ix]
    xlim = [column_descriptions[ix][ivmin], column_descriptions[ix][ivmax]]
    ylabel = p.independent_variable_labels[iy]
    ylim = [column_descriptions[iy][ivmin], column_descriptions[iy][ivmax]]
    zlabel = varlabel
    zlim = [0.0, 3.0]  # <HACK/>
    training_point_color = 'black'
    figsize = (19.2, 4.8)  # For row of 3 contour plots.
    nrows, ncols = 1, 3
    plot_filename = f"{varname}.png"

    if verbose:
        print(f"Creating predicted/empirical/error figure for {varname}.")

    # Create the figure.
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Comparison of trained and empirical solutions for {varlabel}"
    )
    gs = mpl.gridspec.GridSpec(nrows, ncols)

    # Create the left plot (trained solution).
    ax = fig.add_subplot(gs[0])
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.grid()
    cnt = ax.contour(X, Y, Z_trained)
    ax.clabel(cnt, inline=True)

    # Create the middle plot (empirical solution).
    ax = fig.add_subplot(gs[1])
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.grid()
    cnt = ax.contour(X, Y, Z_empirical)
    ax.clabel(cnt, inline=True)

    # Create the right plot (error).
    ax = fig.add_subplot(gs[2])
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.grid()
    cnt = ax.contour(X, Y, Z_error)
    ax.clabel(cnt, inline=True)

    # Add a plot title with the RMS error.
    text = f"RMS error = {z_rmserr:.2E}"
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
