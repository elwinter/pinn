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
description = "Solve a set of coupled 1st-order PDE using the PINN method."

# Program defaults

# Default activation function to use in hidden nodes.
default_activation = "sigmoid"

# Default batch size for training points.
DEFAULT_BATCH_SIZE = 1000

# Default learning rate.
default_learning_rate = 0.01

# Default maximum number of training epochs.
default_max_epochs = 100

# Default number of hidden nodes per layer.
default_n_hid = 10

# Default number of layers in the fully-connected network, each with n_hid
# nodes.
default_n_layers = 1

# Default TensorFlow precision for computations.
default_precision = "float32"

# Default random number generator seed.
default_seed = 0

# Default absolute tolerance for consecutive loss function values to indicate
# convergence.
default_tolerance = 1e-6

# Default normalized weight to apply to the boundary condition loss function.
default_w_data = 0.0


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
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        "-a", "--activation", default=default_activation,
        help="Specify activation function (default: %(default)s)."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Batch size for training points (default: %(default)s)."
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
        "--learning_rate", type=float, default=default_learning_rate,
        help="Learning rate for training (default: %(default)s)"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=default_max_epochs,
        help="Maximum number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--n_hid", type=int, default=default_n_hid,
        help="Number of hidden nodes per layer (default: %(default)s)"
    )
    parser.add_argument(
        "--n_layers", type=int, default=default_n_layers,
        help="Number of hidden layers (default: %(default)s)"
    )
    parser.add_argument(
        "--precision", type=str, default=default_precision,
        help="Precision to use in TensorFlow solution (default: %(default)s)"
    )
    parser.add_argument(
        "--save_model", action="store_true",
        help="Save the trained model (default: %(default)s)."
    )
    parser.add_argument(
        "--save_weights", action="store_true",
        help="Save the model weights at each epoch (default: %(default)s)."
    )
    parser.add_argument(
        "--seed", type=int, default=default_seed,
        help="Seed for random number generator (default: %(default)s)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=default_tolerance,
        help="Absolute loss function convergence tolerance "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "-w", "--w_data", type=float, default=default_w_data,
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
    batch_size = args.batch_size
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
    try:
        os.mkdir(output_dir)
    except:
        pass  # Overwrite existing directory.

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

    # Read the known data, including boundary conditions.
    if verbose:
        print("Reading known data from %s." % data_points_path)
    # Shape is (n_data, n_dim + n_var)
    XY_data = np.loadtxt(data_points_path, dtype=precision)
    # If the data shape is 1-D (only one dimension), reshape to 2-D,
    # to make compatible with later calls.
    if len(XY_data.shape) == 1:
        XY_data = XY_data.reshape(1, XY_data.shape[0])
    np.savetxt(os.path.join(output_dir, "XY_data.dat"), XY_data)

    # Count the data points.
    n_data = XY_data.shape[0]
    if debug:
        print("n_data = %s" % n_data)

    # Compute the normalized weight for the equation residuals.
    w_res = 1.0 - w_data
    if debug:
        print("w_data = %s" % w_data)
        print("w_res = %s" % w_res)

    # Build a model with one output for each dependent variable.
    if verbose:
        print("Creating neural network.")
    model = common.build_multi_output_model(p.n_dim, p.n_var, n_layers, H, activation)
    if verbose:
        print(model.summary())

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

    # Extract the values of the supplied data points.
    # Shape (n_data, n_var)
    Y_data0 = tf.Variable(XY_data[:, p.n_dim:], dtype=precision)

    # Clear the convergence flag to start.
    converged = False

    # Train the models.

    # Record the training start time.
    t_start = datetime.datetime.now()
    if verbose:
        print("Training started at", t_start)

    # Compute the number of batches.
    n_batch = int(np.ceil(n_train/batch_size))

    # Print full arrays.
    np.set_printoptions(threshold=np.inf)

    for epoch in range(max_epochs):
        print("*************************************************************")
        print("Starting epoch %d." % epoch)

        # Run the forward pass.
        # tape0 is for computing gradients wrt network parameters.
        # tape1 is for computing 1st-order derivatives of outputs wrt inputs.
        with tf.GradientTape(persistent=True) as tape0:
            with tf.GradientTape(persistent=True) as tape1:
                Y_batches = []
                gradY_batches = []
                for i_batch in range(n_batch):
                    print("Starting batch %s." % i_batch)

                    # Compute the network outputs at batch training points.
                    # Y_batch is a tf.Tensor of shape (batch_size, p.n_var).
                    i_start = i_batch*batch_size
                    i_stop = (i_batch + 1)*batch_size
                    X_batch = X_train[i_start:i_stop, :]
                    Y_batch = model(X_batch)
                    Y_batches.append(Y_batch)

                    # Compute the network outputs at the data points.
                    # Y_data is a tf.Tensor of shape (n_data, p.n_var).
                    # print("X_data = %s" % X_data)
                    # Y_data = model(X_data)
                    # print("Y_data = %s" % Y_data)

                    # Compute the jacobian of the network outputs wrt inputs at batch points.
                    jacY_batch = tape1.jacobian(Y_batch, X_batch)

                    # Extract the gradients.
                    gradY_batch = tf.stack(
                        [tf.stack(
                            [tf.linalg.diag_part(jacY_batch[:, i, :, j])
                            for j in range(p.n_dim)], axis=1)
                            for i in range(p.n_var)], axis=1)
                    gradY_batches.append(gradY_batch)

                # Convert the batched results to whole-training-set results.
                Y_train = tf.reshape(tf.stack(Y_batches), (n_train, p.n_var))
                gradY_train = tf.reshape(tf.stack(gradY_batches), (n_train, p.n_var, p.n_dim))

                # Compute the estimates of the differential equations at all
                # training points.
                # G_train is a Tensor of shape (n_train, p.n_var).
                G_train = tf.stack([f(X_train, Y_train, gradY_train) for f in p.de], axis=1)
                # print("G_train = %s" % G_train)

    #         # The loss function is composed of 2 terms:
    #         # L = (1 - w_data)*L_res + w_data*L_dat
    #         # where:
    #         # L = total loss function
    #         # L_res = loss function from equation residuals at training points
    #         # L_dat = loss function for data residuals at data points
    #         # Similarly, for each modeled variable m, there is the per-variable loss:
    #         # Lm = (1 - w_data)*Lm_res + w_data*Lm_dat
    #         # Therefore (sums over p.nvar):
    #         # L = SUM(Lm)
    #         # L_res = SUM(Lm_res)
    #         # L_dat = SUM(Lm_dat)

    #         # Compute the loss function for the equation residuals for each variable,
    #         # and the total.
    #         # Tensor, shape (p.n_var,).
    #         Lm_res = tf.math.sqrt(tf.reduce_sum(G_train**2, axis=0)/n_train)
    #         # Tensor, shape () (scalar)
    #         L_res = tf.reduce_sum(Lm_res)

    #         # Compute the loss function for the data residuals for each variable,
    #         # and the total.
    #         # Tensor, shape (n_data, p.n_var)
    #         E_data = Y_data - Y_data0
    #         # Tensor, shape (p.n_var,)
    #         Lm_data = tf.math.sqrt(tf.reduce_sum(E_data**2, axis=0)/n_data)
    #         # Tensor, shape () (scalar)
    #         L_data = tf.reduce_sum(Lm_data)

    #         # Compute the weighted losses per variable, and overall.
    #         # Tensor, shape (n_var,)
    #         Lm = w_res*Lm_res + w_data*Lm_data
    #         # Tensor, shape () (scalar)
    #         L = w_res*L_res + w_data*L_data

    #     # Save the current losses.
    #     losses_model_res.append(Lm_res.numpy())
    #     losses_model_data.append(Lm_data.numpy())
    #     losses_model.append(Lm.numpy())
    #     losses_res.append(L_res.numpy())
    #     losses_data.append(L_data.numpy())
    #     losses.append(L.numpy())

    #     # Save the current model weights.
    #     if save_weights:
    #         model.save_weights(
    #             os.path.join(output_dir, "weights", "weights_%06d" % epoch)
    #         )

    #     # Check for convergence.
    #     if convcheck:
    #         if epoch > 1:
    #             loss_delta = losses[-1] - losses[-2]
    #             if abs(loss_delta) <= tol:
    #                 converged = True
    #                 break

    #     # Compute the gradient of the loss function wrt the network parameters.
    #     # pgrad is a list of lists of Tensor objects.
    #     # There are p.n_var sub-lists in the top-level list.
    #     # There are 3 Tensors in each sub-list, with shapes:
    #     # Input weights: (H, p.n_dim)
    #     # Input biases: (H,)
    #     # Output weights: (H, 1)
    #     # Each Tensor is shaped based on model.trainable_variables.
    #     pgrad = tape0.gradient(L, model.trainable_variables)

    #     # Update the parameters for this epoch.
    #     optimizer.apply_gradients(zip(pgrad, model.trainable_variables))

    #     if verbose and epoch % 1 == 0:
    #         print("Ending epoch %s, (L, L_res, L_data) = (%e, %e, %e)" %
    #               (epoch, L.numpy(), L_res.numpy(), L_data.numpy()))

    # # Count the last epoch.
    # n_epochs = epoch + 1

    # # Record the training end time.
    # t_stop = datetime.datetime.now()
    # t_elapsed = t_stop - t_start
    # if verbose:
    #     print("Training stopped at", t_stop)
    #     print(
    #         "Total training time was %s seconds." % t_elapsed.total_seconds()
    #     )
    #     print("Epochs: %d" % n_epochs)
    #     print("Final value of loss function: %f" % L)
    #     print("converged = %s" % converged)

    # # Convert the loss function histories to numpy arrays.
    # losses_model_res = np.array(losses_model_res)
    # losses_model_data = np.array(losses_model_data)
    # losses_model = np.array(losses_model)
    # losses_res = np.array(losses_res)
    # losses_data = np.array(losses_data)
    # losses = np.array(losses)

    # # Save the loss function histories.
    # if verbose:
    #     print("Saving loss function histories.")
    # np.savetxt(os.path.join(output_dir, 'losses_model_res.dat'),
    #            losses_model_res)
    # np.savetxt(os.path.join(output_dir, 'losses_model_data.dat'),
    #            losses_model_data)
    # np.savetxt(os.path.join(output_dir, 'losses_model.dat'), losses_model)
    # np.savetxt(os.path.join(output_dir, 'losses_res.dat'), losses_res)
    # np.savetxt(os.path.join(output_dir, 'losses_data.dat'), losses_data)
    # np.savetxt(os.path.join(output_dir, 'losses.dat'), losses)

    # # Compute and save the trained results at training points.
    # if verbose:
    #     print("Computing and saving trained results.")
    # with tf.GradientTape(persistent=True) as tape1:
    #     Y_train = model(X_train)
    # np.savetxt(os.path.join(output_dir, "Y_train.dat"), Y_train.numpy())

    # # Save the trained models.
    # if save_model:
    #     model.save(os.path.join(output_dir, "models"))


if __name__ == "__main__":
    """Begin main program."""
    main()
