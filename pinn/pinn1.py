#!/usr/bin/env python

"""Use a set of neural networks to solve a set of coupled 1st-order PDE BVP.

This program will use a set of neural networks to solve a set of coupled
1st-order PDEs as a BVP.

The values of the independent variables used in the training points are
stored in the array X, of shape (n_train, n_dim), where n_train is the
number of training points, and n_dim is the number of dimensions (independent
variables).

The values of the dependent variables are stored in the array Y,
of shape (n_train, n_var), where n_var is the number of dependent variables.

The first derivatives of each Y with respect to each independent variable are
stored in the array delY, shape (n_train, n_var, n_dim).

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
DESCRIPTION = "Solve a set of 1st-order DE using the PINN method."

# Program defaults

# Default activation function to use in hidden nodes.
DEFAULT_ACTIVATION = "sigmoid"

# Default number of training samples in batch.
DEFAULT_BATCH_SIZE = -1

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

# Default normalized weight to apply to the boundary condition loss function.
DEFAULT_W_DATA = 0.0


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
    parser = argparse.ArgumentParser(DESCRIPTION)
    parser.add_argument(
        "-a", "--activation", default=DEFAULT_ACTIVATION,
        help="Specify activation function (default: %(default)s)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Size of training batches ({DEFAULT_BATCH_SIZE} "
        "for single batch)  (default: %(default)s)"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to optional input data file (default: %(default)s)."
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
        "--precision", type=str, default=DEFAULT_PRECISION,
        help="Precision to use in TensorFlow solution (default: %(default)s)"
    )
    parser.add_argument(
        "--save_model", type=int, default=DEFAULT_SAVE_MODEL,
        help="Save interval (epochs) for trained model (default: %(default)s)."
        " 0 = do not save, -1 = save at end, n > 0 = save every n epochs."
    )
    parser.add_argument(
        "--save_weights", action="store_true",
        help="Save the model weights at each epoch (default: %(default)s)."
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Seed for random number generator (default: %(default)s)"
    )
    parser.add_argument(
        "--validation", default=None,
        help="Path to optional validation point file (default: %(default)s)."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "-w", "--w_data", type=float, default=DEFAULT_W_DATA,
        help="Normalized weight for data loss function "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "problem",
        help="Path to problem description file."
    )
    parser.add_argument(
        "training_points",
        help="Path to file containing training points."
    )
    return parser


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

    Raises
    ------
    None
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
    batch_size = args.batch_size
    debug = args.debug
    data = args.data
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    H = args.n_hid
    n_layers = args.n_layers
    precision = args.precision
    save_model = args.save_model
    save_weights = args.save_weights
    seed = args.seed
    validation = args.validation
    verbose = args.verbose
    w_data = args.w_data
    problem_path = args.problem
    training_points = args.training_points
    if debug:
        print(f"args = {args}")

    # Set the backend TensorFlow precision.
    if verbose:
        print(f"Setting TensorFlow precision to {precision}.")
    tf.keras.backend.set_floatx(precision)

    # Import the problem to solve.
    if verbose:
        print(f"Importing module for problem {problem_path}.")
    p = import_problem(problem_path)
    if debug:
        print(f"p = {p}")

    # Set up the output directory under the current directory.
    # An exception is raised if the directory already exists.
    output_dir = os.path.join(".", p.__name__)
    if debug:
        print(f"output_dir = {output_dir}")
    # os.mkdir(output_dir)

    # Record system information, network parameters, and problem definition.
    if verbose:
        print("Recording system information, model hyperparameters, and "
              "problem definition.")
    common.save_system_information(output_dir)
    common.save_hyperparameters(args, output_dir)
    common.save_problem_definition(p, output_dir)

    # Read the training points.
    if verbose:
        print(f"Reading training points from {training_points}.")
    # X_train is np.ndarray of shape (n_train, n_dim) OR (n_train,)
    X_train = np.loadtxt(training_points, dtype=precision)
    if debug:
        print(f"X_train = {X_train}")
    # If the data shape is 1-D (only one dimension), reshape to 2-D,
    # to make compatible with later calls.
    if len(X_train.shape) == 1:
        X_train = X_train.reshape(X_train.shape[0], 1)
        if debug:
            print(f"Reshaped X_train = {X_train}")
    np.savetxt(os.path.join(output_dir, "X_train.dat"), X_train)

    # Count the training points.
    n_train = X_train.shape[0]
    if debug:
        print(f"n_train = {n_train}")

    # Determine the number of problem dimensions.
    n_dim = X_train.shape[1]
    if debug:
        print(f"n_dim = {n_dim}")

    # If provided, read and count the additional training data, including
    # boundary conditions.
    if data:
        if verbose:
            print(f"Reading additional training data from {data}.")
        # Shape is (n_data, n_dim + n_var)
        XY_data = np.loadtxt(data, dtype=precision)
        if debug:
            print(f"XY_data = {XY_data}")
        # If the data shape is 1-D (only one data point), reshape to 2-D,
        # to make compatible with later calls.
        if len(XY_data.shape) == 1:
            XY_data = XY_data.reshape(1, XY_data.shape[0])
            if debug:
                print(f"Reshaped XY_data = {XY_data}")
        np.savetxt(os.path.join(output_dir, "XY_data.dat"), XY_data)
        # Extract the locations of the supplied data points.
        # Shape (n_data, n_dim)
        X_data = XY_data[:, :p.n_dim]
        if debug:
            print(f"X_data = {X_data}")
        n_data = XY_data.shape[0]
        if debug:
            print(f"n_data = {n_data}")

    # If provided, read and count the validation points.
    if validation:
        if verbose:
            print(f"Reading validation points from {validation}.")
        # Shape is (n, n_dim)
        X_val = np.loadtxt(validation, dtype=precision)
        if debug:
            print(f"X_val = {X_val}")
        # If the data shape is 1-D (only one dimension), reshape to 2-D,
        # to make compatible with later calls.
        if len(X_val.shape) == 1:
            X_val = X_val.reshape(X_val.shape[0], 1)
            if debug:
                print(f"Reshaped X_val = {X_val}")
        np.savetxt(os.path.join(output_dir, "X_val.dat"), X_val)
        n_val = X_val.shape[0]

    # Compute the normalized weight for the equation residuals, based on the
    # value of the data weight.
    w_res = 1.0 - w_data
    if debug:
        print(f"w_res = {w_res}")
        print(f"w_data = {w_data}")

    # Build one model for each differential equation defined in the problem.
    if verbose:
        print("Creating neural networks models.")
    models = []
    for i in range(p.n_var):
        if verbose:
            print(f"Creating model for {p.dependent_variable_names[i]}.")
        model = common.build_model(n_layers, H, activation)
        models.append(model)
    if debug:
        print(f"models = {models}")

    # Create the optimizer.
    if verbose:
        print("Creating Adam optimizer.")
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    if debug:
        print(f"optimizer = {optimizer}")

    # Set the random number seed for reproducibility.
    if verbose:
        print(f"Seeding random number generator with {seed}.")
    tf.random.set_seed(seed)

    # If a single bach was requested, compute the batch size.
    if batch_size == -1:
        batch_size = n_train
        if debug:
            print(f"Computed batch_size = {batch_size}")

    # Split the training data into randomized batches.
    # Convert each batch to a tf.Variable for use in gradients.
    # I verified separately that this algorithm results in the original
    # training set being equally distributed among the batches.
    batches = []
    training_point_indices = np.arange(n_train)
    np.random.shuffle(training_point_indices)
    i_start = i_stop = 0
    i_batch = 0
    while i_stop < n_train:
        i_stop = i_start + batch_size
        if i_stop > n_train:
            i_stop = n_train
        if verbose:
            print(f"Creating training batch {i_batch} from "
                  f"{i_start} to {i_stop}.")
        batch_indices = training_point_indices[i_start:i_stop]
        batch_points = X_train[batch_indices]
        path = os.path.join(output_dir, f"batch_{i_batch:04d}.dat")
        np.savetxt(path, batch_points)
        batches.append(tf.Variable(batch_points))
        i_start = i_stop
        i_batch += 1

    # Convert additional data locations to tf.Variable.
    if data:
        X_data = tf.Variable(X_data)

    # Convert validation locations to tf.Variable.
    if validation:
        X_val = tf.Variable(X_val)

    # Record the training start time.
    t_start = datetime.datetime.now()
    if verbose:
        print(f"Training started at {t_start}.")

    for epoch in range(max_epochs):
        if debug:
            print(f"Starting epoch {epoch}.")

        # Step 1: Train using the residuals at each training point.
        for (i_batch, X_batch) in enumerate(batches):
            if debug:
                print(f"Starting training with batch {i_batch}.")

            # Determine the length of this batch.
            n_batch = X_batch.shape[0]
            if debug:
                print(f"n_batch = {n_batch}")

            # Run the forward pass for this batch.
            # tape0 is for computing gradients wrt network parameters.
            # tape1 is for computing 1st-order derivatives of outputs wrt
            # inputs.
            with tf.GradientTape(persistent=True) as tape0:
                with tf.GradientTape(persistent=True) as tape1:

                    # Compute the network outputs at all batch points.
                    # Y_batch is a list of tf.Tensor objects.
                    # There are p.n_var Tensors in the list (one per model).
                    # Each Tensor has shape (n_batch, 1).
                    Y_batch = [model(X_batch) for model in models]
                    if debug:
                        print(f"Y_batch = {Y_batch}")

                # Compute the gradients of the network outputs wrt inputs for
                # this batch.
                # dY_dX_batch is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (n_batch, p.n_dim).
                dY_dX_batch = [tape1.gradient(Y, X_batch) for Y in Y_batch]
                if debug:
                    print(f"dY_dX_batch = {dY_dX_batch}")

                # Compute the values of the differential equations at all
                # batch points.
                # G_batch is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (n_batch, 1).
                G_batch = [f(X_batch, Y_batch, dY_dX_batch) for f in p.de]
                if debug:
                    print(f"G_batch = {G_batch}")

                # Compute the loss function for the equation residuals at the
                # batch training points for each model.
                # loss_model_residual_batch is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape () (scalar).
                loss_model_residual_batch = [
                    tf.math.sqrt(tf.reduce_sum(G**2)/n_batch) for G in G_batch
                ]
                if debug:
                    print(f"loss_model_residual_batch = "
                          f"{loss_model_residual_batch}")

                # Compute the aggregate loss over all models.
                L_batch = w_res*tf.math.reduce_sum(loss_model_residual_batch)
                if debug:
                    print(f"L_batch = {L_batch}")

            # Compute the gradient of the loss function wrt the network
            # parameters.
            # pgrad is a list of lists of Tensor objects.
            # There are p.n_var sub-lists in the top-level list.
            # There are 3 Tensors in each sub-list, with shapes:
            # Input weights: (H, p.n_dim)
            # Input biases: (H,)
            # Output weights: (H, 1)
            # Each Tensor is shaped based on model.trainable_variables.
            pgrad_batch = [
                tape0.gradient(L_batch, model.trainable_variables)
                for model in models
            ]
            if debug:
                print(f"pgrad_batch = {pgrad_batch}")

            # Update the parameters for this epoch.
            for (g, m) in zip(pgrad_batch, models):
                optimizer.apply_gradients(zip(g, m.trainable_variables))

        # Step 2: Train using any additional data points.
        if data:

            # Run the forward pass for the data points in a single batch.
            # tape0 is for computing gradients wrt network parameters.
            with tf.GradientTape(persistent=True) as tape0:

                # Compute the network outputs at the data points.
                # Y_data is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (n_data, 1).
                Y_data = [model(X_data) for model in models]

                # Compute the errors for the data points for each model.
                # Em_data is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list.
                # Each Tensor has shape (n_data, 1).
                Em_data = [
                    Y_data[i] - tf.reshape(XY_data[:, p.n_dim + i], (n_data, 1))
                    for i in range(p.n_var)
                ]

                # Compute the loss functions for the data points for each
                # model.
                # Lm_data is a list of Tensor objects.
                # There are p.n_var Tensors in the list.
                # Each Tensor has shape () (scalar).
                Lm_data = [
                    tf.math.sqrt(tf.reduce_sum(E**2)/n_data)
                    for E in Em_data
                ]

                # Compute the total data loss function.
                L_data = w_data*tf.reduce_sum(Lm_data)

            # Compute the gradient of the data loss wrt the network parameters.
            # pgrad_data is a list of lists of Tensor objects.
            # There are p.n_var sub-lists in the top-level list.
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
                print(f"pgrad_data = {pgrad_data}")

            # Update the parameters for this data.
            for (g, m) in zip(pgrad_data, models):
                optimizer.apply_gradients(zip(g, m.trainable_variables))

        if verbose:
            print(f"epoch = {epoch}, L_batch = {L_batch}, L_data = {L_data}")

        if debug:
            print(f"Ending epoch {epoch}.")

    # Count the last epoch.
    n_epochs = epoch + 1
    if debug:
        print(f"n_epochs = {n_epochs}")

    # Record the training end time.
    t_stop = datetime.datetime.now()
    t_elapsed = t_stop - t_start
    if verbose:
        print(f"Training stopped at {t_stop}.")
        print(f"Total training time: {t_elapsed.total_seconds()} seconds")
        print(f"Epochs: {n_epochs}")
        # print(f"Final value of loss function: {loss_residual_batch}")

    # Compute and save the trained results at training points.
    if verbose:
        print("Computing and saving trained results.")
    # Shapes are (n_train, 1)
    with tf.GradientTape(persistent=True) as tape1:
        Y_train = [model(X_train) for model in models]
    # Shapes are (n_train, n_dim)
    # dY_dX_train = [tape1.gradient(Y, X_train) for Y in Y_train]
    for i in range(p.n_var):
        np.savetxt(os.path.join(output_dir, "%s_train.dat" %
                   p.dependent_variable_names[i]),
                   tf.reshape(Y_train[i], (n_train,)))
        # np.savetxt(os.path.join(output_dir, "del_%s_train.dat" %
        #            p.dependent_variable_names[i]), dY_dX_train[i])


if __name__ == "__main__":
    """Begin main program."""
    main()
