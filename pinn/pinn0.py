#!/usr/bin/env python

"""Use PINNs to approximate a multivariable function.

Use neural networks to approximate a multivariable function.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard Python modules.
import argparse
import datetime
import importlib.util
import os
import sys

# Import 3rd-party modules.
import numpy as np
import tensorflow as tf

# Import project modules.
from pinn import common


# Program constants

# Program description.
DESCRIPTION = "Use a neural network to approximate a function."

# Program defaults

# Default activation function to use in hidden nodes.
DEFAULT_ACTIVATION = "sigmoid"

# Default learning rate.
DEFAULT_LEARNING_RATE = 0.01

# Default maximum number of training epochs.
DEFAULT_MAX_EPOCHS = 100

# Default number of hidden nodes per layer.
DEFAULT_N_HID = 10

# Default number of layers in the fully-connected network, each with n_hid
# nodes.
DEFAULT_N_LAYERS = 1

# Default TensorFlow precision for computations.
DEFAULT_PRECISION = "float32"

# Default interval (in epochs) for saving the model.
# 0 = do not save model
# -1 = only save at end
# n > 0: Save after every n epochs.
DEFAULT_SAVE_MODEL = -1

# Default random number generator seed.
DEFAULT_SEED = 0


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
        "-a", "--activation", default=DEFAULT_ACTIVATION,
        help="Specify activation function (default: %(default)s)."
    )
    parser.add_argument(
        "--clobber", action="store_true",
        help="Clobber existing output directory (default: %(default)s)."
    )
    parser.add_argument(
        "-d", "--debug", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
        help="Learning rate for training (default: %(default)s)"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS,
        help="Maximum number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--n_hid", type=int, default=DEFAULT_N_HID,
        help="Number of hidden nodes per layer (default: %(default)s)"
    )
    parser.add_argument(
        "--n_layers", type=int, default=DEFAULT_N_LAYERS,
        help="Number of hidden layers (default: %(default)s)"
    )
    parser.add_argument(
        "--nogpu", action="store_true",
        help="Disable TensorFlow use of GPU(s) (default: %(default)s)."
    )
    parser.add_argument(
        "--precision", type=str, default=DEFAULT_PRECISION,
        help="Precision to use in TensorFlow solution (default: %(default)s)"
    )
    parser.add_argument(
        "--save_model", type=int, default=DEFAULT_SAVE_MODEL,
        help="Save interval (epochs) for trained model (default: %(default)s)."
        " 0 = do not save, -1 = save at end, n > 0 = save every n epochs."
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Seed for random number generator (default: %(default)s)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "--validation", default=None,
        help="Path to optional validation point file (default: %(default)s)."
    )
    parser.add_argument(
        "problem_path",
        help="Path to problem description file."
    )
    parser.add_argument(
        "data",
        help="Path to file containing data points to fit."
    )
    return parser


def disable_gpus():
    """Tell TensorFlow not to use GPU.

    Tell TensorFlow not to use GPU.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError : If this code cannot disable a GPU.
    """
    # Disable all GPUS.
    tf.config.set_visible_devices([], "GPU")

    # Make sure the GPU were disabled.
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"


def import_problem(problem_path):
    """Import the Python file which defines the problem to solve.

    Import the Python file which defines the problem to solve.

    Parameters
    ----------
    problem_path : str
        Path to problem definition file.

    Returns
    -------
    p : module
        Module object for problem definition.
    """
    problem_name = os.path.splitext(os.path.split(problem_path)[-1])[-2]
    spec = importlib.util.spec_from_file_location(problem_name, problem_path)
    p = importlib.util.module_from_spec(spec)
    sys.modules[problem_name] = p
    spec.loader.exec_module(p)
    return p


def main():
    """Begin main program."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    activation = args.activation
    clobber = args.clobber
    debug = args.debug
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    H = args.n_hid
    n_layers = args.n_layers
    nogpu = args.nogpu
    precision = args.precision
    save_model = args.save_model
    seed = args.seed
    verbose = args.verbose
    validation = args.validation
    problem_path = args.problem_path
    data = args.data
    if debug:
        print(f"args = {args}", flush=True)

    # If requested, disable TensorFlow use of GPU.
    if nogpu:
        if verbose:
            print("Disabling TensorFlow use of GPU.", flush=True)
        disable_gpus()

    # Set the backend TensorFlow precision.
    if verbose:
        print(f"Setting TensorFlow precision to {precision}.", flush=True)
    tf.keras.backend.set_floatx(precision)

    # Import the problem to solve.
    if verbose:
        print(f"Importing module for problem {problem_path}.", flush=True)
    p = import_problem(problem_path)
    if debug:
        print(f"p = {p}", flush=True)

    # Set up the output directory under the current directory.
    # An exception is raised if the directory already exists.
    output_dir = os.path.join(".", "pinn0_output")
    if debug:
        print(f"output_dir = {output_dir}", flush=True)
    if os.path.exists(output_dir):
        if not clobber:
            raise TypeError(f"Output directory {output_dir} exists!")
    else:
        os.mkdir(output_dir)

    # Record system information, network parameters, and problem definition.
    if verbose:
        print("Recording system information, model hyperparameters, and "
              "problem definition.", flush=True)
    common.save_system_information(output_dir)
    # common.save_hyperparameters(args, output_dir)
    common.save_problem_definition(p, output_dir)

    # If provided, read and count the additional training data, including
    # boundary conditions.
    if data:
        if verbose:
            print(f"Reading additional training data from {data}.", flush=True)
        # Shape is (n_data, p.n_dim + p.n_var)
        XY_data = np.loadtxt(data, dtype=precision)
        if debug:
            print(f"XY_data = {XY_data}", flush=True)
        # If the data shape is 1-D (only one data point), reshape to 2-D,
        # to make compatible with later calls.
        if len(XY_data.shape) == 1:
            XY_data = XY_data.reshape(1, XY_data.shape[0])
            if debug:
                print(f"Reshaped XY_data = {XY_data}", flush=True)
        np.savetxt(os.path.join(output_dir, "XY_data.dat"), XY_data)
        # Extract the locations of the supplied data points.
        # Shape (n_data, p.n_dim)
        X_data = XY_data[:, :p.n_dim]
        if debug:
            print(f"X_data = {X_data}", flush=True)
        n_data = XY_data.shape[0]
        if debug:
            print(f"n_data = {n_data}", flush=True)

    # Build one model for each dependent variable in the problem.
    if verbose:
        print("Creating neural network models.", flush=True)
    models = []
    for i in range(p.n_var):
        if verbose:
            print(f"Creating model for {p.dependent_variable_names[i]}.",
                  flush=True)
        model = common.build_model(n_layers, H, activation)
        models.append(model)
    if debug:
        print(f"models = {models}", flush=True)

    # Create the optimizer.
    if verbose:
        print("Creating Adam optimizer.", flush=True)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    if debug:
        print(f"optimizer = {optimizer}", flush=True)

    # Set the random number seed for reproducibility.
    if verbose:
        print(f"Seeding random number generator with {seed}.", flush=True)
    tf.random.set_seed(seed)

    # Convert additional data locations to tf.Variable.
    if data:
        X_data = tf.Variable(X_data)
        if debug:
            print(f"TF Variable of X_data = {X_data}", flush=True)

    # Create loss history variables.
    loss = {}
    loss["data"] = []

    # Record the training start time.
    t_start = datetime.datetime.now()
    if verbose:
        print(f"Training started at {t_start}.", flush=True)

    for epoch in range(max_epochs):

        # Run the forward pass for the data points in a single batch.
        # tape0 is for computing gradients wrt network parameters.
        with tf.GradientTape(persistent=True) as tape0:

            # Compute the network outputs at the data points.
            # Y_data is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape (n_data, 1).
            Y_data = [model(X_data) for model in models]
            if debug:
                print(f"Y_data = {Y_data}", flush=True)

            # Compute the errors for the data points for each model.
            # Em_data is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_data, 1).
            Em_data = [
                Y_data[i] - tf.reshape(XY_data[:, p.n_dim + i],
                                        (n_data, 1))
                for i in range(p.n_var)
            ]
            if debug:
                print(f"Em_data = {Em_data}", flush=True)

            # Compute the loss functions for the data points for each
            # model.
            # Lm_data is a list of Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape () (scalar).
            Lm_data = [
                tf.math.sqrt(tf.reduce_sum(E**2)/n_data)
                for E in Em_data
            ]
            if debug:
                print(f"Lm_data = {Lm_data}", flush=True)

            # Compute the data loss function.
            L_data = tf.reduce_sum(Lm_data)
            if debug:
                print(f"L_data = {L_data}", flush=True)

        # Compute the gradient of the data loss wrt the network
        # parameters.
        # pgrad_data is a list of lists of Tensor objects.
        # There are p.n_var sub-lists in the top-level list (one per
        # model).
        # There are 3 Tensors in each sub-list, with shapes:
        # Input weights: (H, p.n_dim)
        # Input biases: (H,)
        # Output weights: (H, 1)
        # Each Tensor is shaped based on model.trainable_variables.
        pgrad_data = [
            tape0.gradient(L_data, model.trainable_variables)
            for model in models
        ]
        if debug:
            print(f"pgrad_data = {pgrad_data}", flush=True)

        # Update the parameters for this data.
        for (g, m) in zip(pgrad_data, models):
            optimizer.apply_gradients(zip(g, m.trainable_variables))

        # Compute the end-of-epoch data loss function.
        Y_data = [model(X_data) for model in models]
        Em_data = [
            Y_data[i] - tf.reshape(XY_data[:, p.n_dim + i], (n_data, 1))
            for i in range(p.n_var)
        ]
        Lm_data = [
            tf.math.sqrt(tf.reduce_sum(E**2)/n_data)
            for E in Em_data
        ]
        L_data = tf.reduce_sum(Lm_data)
        if debug:
            print(f"L_data = {L_data}", flush=True)
        loss["data"].append(L_data)

        if verbose:
            print(f"epoch = {epoch}, L_data = {L_data}", flush=True)

        # Save the trained models.
        if save_model > 0 and epoch % save_model == 0:
            for (i, model) in enumerate(models):
                path = os.path.join(
                    output_dir, "models", f"{epoch}",
                    f"model_{p.dependent_variable_names[i]}"
                )
                model.save(path)

        if debug:
            print(f"Ending epoch {epoch}.", flush=True)

    # Count the last epoch.
    n_epochs = epoch + 1
    if debug:
        print(f"n_epochs = {n_epochs}", flush=True)

    # Record the training end time.
    t_stop = datetime.datetime.now()
    t_elapsed = t_stop - t_start
    if verbose:
        print(f"Training stopped at {t_stop}.", flush=True)
        print(f"Total training time: {t_elapsed.total_seconds()} seconds",
              flush=True)
        print(f"Epochs: {n_epochs}", flush=True)
        print(f"Final value of loss function: {L_data}", flush=True)

    # Save the loss histories.
    np.savetxt(os.path.join(output_dir, "L_data.dat"), loss["data"])

    # Save the trained models.
    if save_model != 0:
        for (i, model) in enumerate(models):
            path = os.path.join(
                output_dir, "models", f"{n_epochs}",
                f"model_{p.dependent_variable_names[i]}"
            )
            model.save(path)


if __name__ == "__main__":
    """Begin main program."""
    main()
