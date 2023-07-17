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
DESCRIPTION = "Solve a set of coupled 1st-order PDE using the PINN method."

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
DEFAULT_SAVE_MODEL = 0

# Default random number generator seed.
DEFAULT_SEED = 0

# Default absolute tolerance for consecutive loss function values to indicate
# convergence.
DEFAULT_TOLERANCE = 1e-6

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
    """
    parser = argparse.ArgumentParser(DESCRIPTION)
    parser.add_argument(
        "-a", "--activation", default=DEFAULT_ACTIVATION,
        help="Specify activation function (default: %(default)s)."
    )
    parser.add_argument(
        "--convcheck", action="store_true",
        help="Perform convergence check (default: %(default)s)."
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
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help="Absolute loss function convergence tolerance "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "--validation_points", default=None,
        help="Path to optional validation point file (default: %(default)s)."
    )
    parser.add_argument(
        "-w", "--w_data", type=float, default=DEFAULT_W_DATA,
        help="Normalized weight for data loss function "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "problem_path",
        help="Path to problem description file."
    )
    parser.add_argument(
        "training_points_path",
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
    convcheck = args.convcheck
    debug = args.debug
    data_points_path = args.data
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    H = args.n_hid
    n_layers = args.n_layers
    precision = args.precision
    save_model = args.save_model
    save_weights = args.save_weights
    seed = args.seed
    tol = args.tolerance
    verbose = args.verbose
    validation_points = args.validation_points
    w_data = args.w_data
    problem_path = args.problem_path
    training_points_path = args.training_points_path
    if debug:
        print("args = %s" % args)

    # Set the backend TensorFlow precision.
    if verbose:
        print("Setting TensorFlow precision to %s." % precision)
    tf.keras.backend.set_floatx(precision)

    # Import the problem to solve.
    if verbose:
        print("Importing module for problem '%s'." % problem_path)
    p = import_problem(problem_path)
    if debug:
        print("p = %s" % p)

    # Set up the output directory under the current directory.
    # An exception is raised if the directory already exists.
    output_dir = os.path.join(".", p.__name__)
    if debug:
        print("output_dir = %s" % output_dir)
    os.mkdir(output_dir)

    # Record system information, network parameters, and problem definition.
    if verbose:
        print("Recording system information, model hyperparameters, and "
              "problem definition.")
    common.save_system_information(output_dir)
    common.save_hyperparameters(args, output_dir)
    common.save_problem_definition(p, output_dir)

    # Read the training points.
    if verbose:
        print("Reading training points from %s." % training_points_path)
    # Shape is (n_train, n_dim)
    X_train = np.loadtxt(training_points_path, dtype=precision)
    # If the data shape is 1-D (only one dimension), reshape to 2-D,
    # to make compatible with later calls.
    if len(X_train.shape) == 1:
        X_train = X_train.reshape(X_train.shape[0], 1)
    np.savetxt(os.path.join(output_dir, "X_train.dat"), X_train)

    # Count the training points.
    n_train = X_train.shape[0]
    if debug:
        print("n_train = %s" % n_train)

    # If provided, read the known data, including boundary conditions.
    if data_points_path:
        if verbose:
            print("Reading known data from %s." % data_points_path)
        # Shape is (n_data, n_dim + n_var)
        XY_data = np.loadtxt(data_points_path, dtype=precision)
        # If the data shape is 1-D (only one dimension), reshape to 2-D,
        # to make compatible with later calls.
        if len(XY_data.shape) == 1:
            XY_data = XY_data.reshape(1, XY_data.shape[0])
        np.savetxt(os.path.join(output_dir, "XY_data.dat"), XY_data)

    # If provided, read the validation points.
    if validation_points:
        if verbose:
            print("Reading validation points from %s." % validation_points)
        # Shape is (n, n_dim)
        X_val = np.loadtxt(validation_points, dtype=precision)
        # If the data shape is 1-D (only one dimension), reshape to 2-D,
        # to make compatible with later calls.
        if len(X_val.shape) == 1:
            X_val = X_val.reshape(1, X_val.shape[0])
        np.savetxt(os.path.join(output_dir, "X_val.dat"), X_val)
        n_val = X_val.shape[0]

    # Count the data points.
    n_data = XY_data.shape[0]
    if debug:
        print("n_data = %s" % n_data)

    # Compute the normalized weight for the equation residuals.
    w_res = 1.0 - w_data
    if debug:
        print("w_data = %s" % w_data)
        print("w_res = %s" % w_res)

    # Build one model for each equation.
    if verbose:
        print("Creating neural networks.")
    models = []
    for i in range(p.n_var):
        if verbose:
            print("Creating neural network for %s." %
                  p.dependent_variable_names[i])
        model = common.build_model(n_layers, H, activation)
        models.append(model)
    if debug:
        print("models = %s" % models)

    # Create the optimizer.
    if verbose:
        print("Creating Adam optimizer.")
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    if debug:
        print("optimizer = %s" % optimizer)

    # Create loss history variables.
    # Loss for each model for equation residuals.
    losses_model_res = []
    # Loss for each model for data points.
    losses_model_data = []
    # Total loss for each model.
    losses_model = []
    # Total loss for all models for equation residuals.
    losses_res = []
    # Total loss for all models for data points.
    losses_data = []
    # Total loss for all models.
    losses = []

    # Set the random number seed for reproducibility.
    if verbose:
        print("Seeding random number generator with %s." % seed)
    tf.random.set_seed(seed)

    # Convert the training data Variables for convenience.
    # The NumPy arrays must be converted to TensorFlow.
    # Shape (n_train, n_dim)
    X_train = tf.Variable(X_train, dtype=precision)
    # Extract the locations of the supplied data points.
    # Shape (n_data, n_dim)
    X_data = tf.Variable(XY_data[:, :p.n_dim], dtype=precision)
    if validation_points:
        X_val = tf.Variable(X_val, dtype=precision)

    # Clear the convergence flag to start.
    converged = False

    # Train the models.

    # Record the training start time.
    t_start = datetime.datetime.now()
    if verbose:
        print("Training started at", t_start)

    for epoch in range(max_epochs):

        # Run the forward pass.
        # tape0 is for computing gradients wrt network parameters.
        # tape1 is for computing 1st-order derivatives of outputs wrt inputs.
        with tf.GradientTape(persistent=True) as tape0:
            with tf.GradientTape(persistent=True) as tape1:

                # Compute the network outputs at all training points.
                # Y is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list.
                # Each Tensor has shape (n_train, 1).
                Y_train = [model(X_train) for model in models]

                # Compute the network outputs at the data points.
                # Y_data is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list.
                # Each Tensor has shape (n_data, 1).
                Y_data = [model(X_data) for model in models]

                # If available, compute the network outputs at all
                # validation points.
                # Y_val is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list.
                # Each Tensor has shape (n_val, 1).
                if validation_points:
                    Y_val = [model(X_val) for model in models]

            # Compute the gradients of the network outputs wrt inputs at all
            # *training* points.
            # dY_dX is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_train, p.n_dim).
            dY_dX_train = [tape1.gradient(Y, X_train) for Y in Y_train]

            # If available, compute the gradients of the network outputs wrt
            # inputs at all *validation* points.
            # dY_dX_val is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_val, p.n_dim).
            if validation_points:
                dY_dX_val = [tape1.gradient(Y_val, X_val) for Y in Y_val]

            # Compute the estimates of the differential equations at all
            # training points.
            # G is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_train, 1).
            G_train = [f(X_train, Y_train, dY_dX_train) for f in p.de]

            # If available, compute the estimates of the differential equations
            # at all validation points.
            # G_val is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_train, 1).
            if validation_points:
                G_val = [f(X_val, Y_val, dY_dX_val) for f in p.de]

            # Compute the loss function for the equation residuals at the
            # training points for each model.
            # Lm_res_train is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape () (scalar).
            Lm_res_train = [
                tf.math.sqrt(tf.reduce_sum(G**2)/n_train) for G in G_train
            ]

            # If available, compute the loss function for the equation residuals
            # at the validation points for each model.
            # Lm_res_val is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape () (scalar).
            if validation_points:
                Lm_res_val = [
                    tf.math.sqrt(tf.reduce_sum(G**2)/n_val) for G in G_val
                ]

            # Compute the errors for the data points for each model.
            # Em_data is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_data, 1).
            Em_data = [
                Y_data[i] - tf.reshape(XY_data[:, p.n_dim + i], (n_data, 1))
                for i in range(p.n_var)
            ]

            # Compute the loss functions for the data points for each model.
            # Lm_data is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape () (scalar).
            Lm_data = [
                tf.math.sqrt(tf.reduce_sum(E**2)/n_data)
                for E in Em_data
            ]

            # Compute the total losses for each model.
            # Lm is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape () (scalar).
            Lm = [
                w_res*loss_res + w_data*loss_data
                for (loss_res, loss_data) in zip(Lm_res_train, Lm_data)
            ]

            # Compute the total loss for all training points for the model
            # collection.
            # Tensor shape () (scalar).
            L_res = tf.math.reduce_sum(Lm_res_train)

            # If available, compute the total loss for all validation points
            # for the model collection.
            # Tensor shape () (scalar).
            if validation_points:
                L_res_val = tf.math.reduce_sum(Lm_res_val)

            # Compute the total loss for data points for the model collection.
            # Tensor shape () (scalar).
            L_data = tf.math.reduce_sum(Lm_data)

            # Compute the total loss for all points for the model collection.
            # Tensor shape () (scalar).
            L = tf.math.reduce_sum(Lm)

        # Save the current losses.
        # The per-model loss histories are lists of lists of Tensors.
        # The top-level list has length n_epochs.
        # Each sub-list has length p.n_var.
        # Each Tensor is shape () (scalar).
        losses_model_res.append(Lm_res_train)
        losses_model_data.append(Lm_data)
        losses_model.append(Lm)
        # The total loss histories are lists of scalars.
        losses_res.append(L_res.numpy())
        losses_data.append(L_data.numpy())
        losses.append(L.numpy())

        # Save the current model weights.
        if save_weights:
            for (i, model) in enumerate(models):
                model.save_weights(
                    os.path.join(
                        output_dir, "weights_" + p.dependent_variable_names[i],
                        "weights_%06d" % epoch
                    )
                )

        # Check for convergence.
        if convcheck:
            if epoch > 1:
                loss_delta = losses[-1] - losses[-2]
                if abs(loss_delta) <= tol:
                    converged = True
                    break

        # Compute the gradient of the loss function wrt the network parameters.
        # pgrad is a list of lists of Tensor objects.
        # There are p.n_var sub-lists in the top-level list.
        # There are 3 Tensors in each sub-list, with shapes:
        # Input weights: (H, p.n_dim)
        # Input biases: (H,)
        # Output weights: (H, 1)
        # Each Tensor is shaped based on model.trainable_variables.
        pgrad = [
            tape0.gradient(L, model.trainable_variables)
            for model in models
        ]

        # Update the parameters for this epoch.
        for (g, m) in zip(pgrad, models):
            optimizer.apply_gradients(zip(g, m.trainable_variables))

        # Save the trained models.
        if save_model > 0 and epoch % save_model == 0:
            for (i, model) in enumerate(models):
                path = os.path.join(
                    output_dir, "models", f"{epoch}",
                    f"model_{p.dependent_variable_names[i]}"
                )
                model.save(path)

        if verbose and epoch % 1 == 0:
            if validation_points:
                print("Ending epoch %s, (L, L_res, L_data, L_res_val) = (%e, %e, %e, %e)" %
                      (epoch, L.numpy(), L_res.numpy(), L_data.numpy(), L_res_val.numpy()))
            else:
                print("Ending epoch %s, (L, L_res, L_data) = (%e, %e, %e)" %
                      (epoch, L.numpy(), L_res.numpy(), L_data.numpy()))

        # Cancel training if NaN is detected in the overall loss function.
        if np.isnan(L):
            if verbose:
                print(f"L = {np.nan} at epoch {epoch}, aborting training.")
            break

    # Count the last epoch.
    n_epochs = epoch + 1

    # Record the training end time.
    t_stop = datetime.datetime.now()
    t_elapsed = t_stop - t_start
    if verbose:
        print("Training stopped at", t_stop)
        print(
            "Total training time was %s seconds." % t_elapsed.total_seconds()
        )
        print("Epochs: %d" % n_epochs)
        print("Final value of loss function: %f" % L)
        print("converged = %s" % converged)

    # Convert the loss function histories to numpy arrays.
    losses_model_res = np.array(losses_model_res)
    losses_model_data = np.array(losses_model_data)
    losses_model = np.array(losses_model)
    losses_res = np.array(losses_res)
    losses_data = np.array(losses_data)
    losses = np.array(losses)

    # Save the loss function histories.
    if verbose:
        print("Saving loss function histories.")
    np.savetxt(os.path.join(output_dir, 'losses_model_res.dat'),
               losses_model_res)
    np.savetxt(os.path.join(output_dir, 'losses_model_data.dat'),
               losses_model_data)
    np.savetxt(os.path.join(output_dir, 'losses_model.dat'), losses_model)
    np.savetxt(os.path.join(output_dir, 'losses_res.dat'), losses_res)
    np.savetxt(os.path.join(output_dir, 'losses_data.dat'), losses_data)
    np.savetxt(os.path.join(output_dir, 'losses.dat'), losses)

    # Compute and save the trained results at training points.
    if verbose:
        print("Computing and saving trained results.")
    # Shapes are (n_train, 1)
    with tf.GradientTape(persistent=True) as tape1:
        Y_train = [model(X_train) for model in models]
    # Shapes are (n_train, n_dim)
    dY_dX_train = [tape1.gradient(Y, X_train) for Y in Y_train]
    for i in range(p.n_var):
        np.savetxt(os.path.join(output_dir, "%s_train.dat" %
                   p.dependent_variable_names[i]),
                   tf.reshape(Y_train[i], (n_train,)))
        np.savetxt(os.path.join(output_dir, "del_%s_train.dat" %
                   p.dependent_variable_names[i]), dY_dX_train[i])

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
