#!/usr/bin/env python

"""Use a neural network to solve a set of coupled 1st-order PDE BVP.

This program will use a neural network to solve a set of coupled 1st-order
PDEs.

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
description = "Solve a set of coupled 1st-order PDE using a multi-output PINN."

# Program defaults

# Default activation function to use in hidden nodes.
default_activation = "sigmoid"

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

# Default normalized weight to apply to the data loss function.
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
        "--convcheck", action="store_true",
        help="Perform convergence check (default: %(default)s)."
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to optional input data file (default: %(default)s)."
    )
    parser.add_argument(
        "-d", "--debug", action="store_true",
        help="Print debugging output (default: %(default)s)."
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
    # Extract the name of the python module.
    problem_name = os.path.splitext(os.path.split(problem_path)[-1])[-2]

    # Create the module spec for the module.
    spec = importlib.util.spec_from_file_location(problem_name, problem_path)

    # Create the module object.
    p = importlib.util.module_from_spec(spec)

    # Save the new module in the list of imported modules.
    sys.modules[problem_name] = p

    # Execute the module.
    spec.loader.exec_module(p)

    # Return the module.
    return p


def main():
    """Begin main program."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    activation = args.activation
    convcheck = args.convcheck
    data_points_path = args.data
    debug = args.debug
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
    # <HACK>
    try:
        os.mkdir(output_dir)
    except:
        pass
    # </HACK>

    # Record system information, network parameters, and problem definition.
    if verbose:
        print("Recording system information, model hyperparameters, and "
              "problem definition.")
    common.save_system_information(output_dir)
    common.save_hyperparameters(args, output_dir)
    common.save_problem_definition(p, output_dir)

    # Read the training points, and save a copy.
    if verbose:
        print("Reading training points from %s." % training_points_path)
    X_train = np.loadtxt(training_points_path, dtype=precision)
    # X_train must be a np.ndarray of shape (n_train, n_dim).
    # If the just-read X_train shape is 1-D (only one dimension), reshape to
    # (n_train, 1) to make compatible with later calls.
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
    # XY_data is a np.ndarray of shape (n_data, n_dim + n_var).
    XY_data = np.loadtxt(data_points_path, dtype=precision)
    np.savetxt(os.path.join(output_dir, "XY_data.dat"), XY_data)

    # Count the data points.
    n_data = XY_data.shape[0]
    if debug:
        print("n_data = %s" % n_data)

    # Compute the normalized weight for the PDE residuals.
    w_res = 1.0 - w_data
    if debug:
        print("w_data = %s" % w_data)
        print("w_res = %s" % w_res)

    # Build a model with one input for each independent variable, and one
    # output for each dependent variable.
    if verbose:
        print("Creating neural network with %s inputs and %s outputs." %
              (p.n_dim, p.n_var))
    model = common.build_multi_output_model(
        p.n_dim, p.n_var, n_layers, H, activation,
    )
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
    losses_variable_res = []
    # Loss for each model for data points.
    losses_variable_data = []
    # Total loss for each model.
    losses_variable = []
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

    # The NumPy arrays must be converted to TensorFlow Variable.
    # Shape (n_train, n_dim)
    X_train = tf.Variable(X_train, dtype=precision)
    # Extract the locations of the supplied data points.
    # Shape (n_data, n_dim)
    X_data = tf.Variable(XY_data[:, :p.n_dim], dtype=precision)
    # Extract the values of the supplied data points.
    # Shape (n_data, n_var)
    Y_data = tf.Variable(XY_data[:, p.n_dim:], dtype=precision)

    # Clear the convergence flag to start.
    converged = False

    # Train the model, proving the entire training and data sets each epoch,
    # as a single batch.

    # Record the training start time.
    t_start = datetime.datetime.now()
    if verbose:
        print("Training started at", t_start)

    # Traing the network for the specified number of epochs.
    for epoch in range(max_epochs):
        if debug:
            print("Starting epoch %s." % epoch)

        # Run the forward pass.
        # tape0 is for computing gradients wrt network parameters.
        # tape1 is for computing 1st-order derivatives of outputs wrt inputs.
        with tf.GradientTape(persistent=True) as tape0:
            with tf.GradientTape(persistent=True) as tape1:

                # Compute the network outputs at all training points.
                # Y_train is a Tensor of shape (n_train, p.n_var).
                Ym_temp = model(X_train)
                if debug:
                    print("Ym_temp = %s" % Ym_temp)

                # Break the training outputs into a list of p.n_var individual
                # Tensor objects. This is required  in order to perform
                # gradients on a per- output-variable basis.
                Ym_train = [Ym_temp[:, i] for i in range(p.n_var)]
                if debug:
                    print("Ym_train = %s" % Ym_train)

                # Compute the network outputs at the data points.
                # Y_data is a Tensor of shape (n_data, p.n_var).
                Ym_data = model(X_data)
                if debug:
                    print("Ym_data = %s" % Ym_data)

            # Compute the gradients of the network outputs wrt inputs at all
            # *training* points.
            # dY_dX_train is a list of p.n_var tf.Tensor objects.
            # Each Tensor has shape (n_train, p.n_dim).
            dY_dX_train = [tape1.gradient(Y, X_train) for Y in Ym_train]

            # Compute the estimates of the differential equations at all
            # training points.
            # G_tmp is a list of p.n_var Tensor objects.
            # Each Tensor has shape (n_train, 1).
            # G_train is a Tensor of shape (n_train, p.n_var).
            G_train = [f(X_train, Ym_train, dY_dX_train) for f in p.de]
            if debug:
                print("G_train = %s" % G_train)

            # Compute the loss function for the PDE residuals at the training
            # points for each variable.
            # Lv_res_train is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape () (scalar).
            Lv_res_train = [
                tf.math.sqrt(tf.reduce_sum(G**2)/n_train) for G in G_train
            ]

            # Compute the errors for the data points for each variable.
            # Ev_data is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_data,).
            Ev_data = [Ym_data[:, i] - Y_data[:, i] for i in range(p.n_var)]

            # Compute the loss functions for the data points for each variable.
            # Lv_data is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape () (scalar).
            Lv_data = [
                tf.math.sqrt(tf.reduce_sum(E**2)/n_data)
                for E in Ev_data
            ]

            # Compute the total losses for each variable.
            # Lv is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape () (scalar).
            Lv = [
                w_res*loss_res + w_data*loss_data
                for (loss_res, loss_data) in zip(Lv_res_train, Lv_data)
            ]

            # Compute the total loss for all training points for the complete
            # model.
            # Tensor shape () (scalar).
            L_res = tf.math.reduce_sum(Lv_res_train)

            # Compute the total loss for data points for the model collection.
            # Tensor shape () (scalar).
            L_data = tf.math.reduce_sum(Lv_data)

            # Compute the total loss for all points for the complete model.
            # Tensor shape () (scalar).
            L = tf.math.reduce_sum(Lv)

        # Save the current losses.
        # The per-model loss histories are lists of lists of Tensors.
        # The top-level list has length n_epochs.
        # Each sub-list has length p.n_var.
        # Each Tensor is shape () (scalar).
        losses_variable_res.append(Lv_res_train)
        losses_variable_data.append(Lv_data)
        losses_variable.append(Lv)
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
        # pgrad is a list of Tensor objects.
        # Each Tensor represents the variables of a layer in the model.
        pgrad = tape0.gradient(L, model.trainable_variables)
        if debug:
            print("pgrad = %s" % pgrad)

        # Update the parameters for this epoch.
        optimizer.apply_gradients(zip(pgrad, model.trainable_variables))

        if verbose and epoch % 1 == 0:
            print("Ending epoch %s, (L, L_res, L_data) = (%e, %e, %e)" %
                  (epoch, L.numpy(), L_res.numpy(), L_data.numpy()))
        if debug:
            print("Ending epoch %s." % epoch)

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
    losses_variable_res = np.array(losses_variable_res)
    losses_variable_data = np.array(losses_variable_data)
    losses_variable = np.array(losses_variable)
    losses_res = np.array(losses_res)
    losses_data = np.array(losses_data)
    losses = np.array(losses)

    # Save the loss function histories.
    if verbose:
        print("Saving loss function histories.")
    np.savetxt(os.path.join(output_dir, 'losses_variable_res.dat'),
               losses_variable_res)
    np.savetxt(os.path.join(output_dir, 'losses_variable_data.dat'),
               losses_variable_data)
    np.savetxt(os.path.join(output_dir, 'losses_variable.dat'), losses_variable)
    np.savetxt(os.path.join(output_dir, 'losses_res.dat'), losses_res)
    np.savetxt(os.path.join(output_dir, 'losses_data.dat'), losses_data)
    np.savetxt(os.path.join(output_dir, 'losses.dat'), losses)

    # Compute and save the trained results at training points.
    if verbose:
        print("Computing and saving trained results.")
    # Shapes are (n_train, 1)
    with tf.GradientTape(persistent=True) as tape1:
        Ym_train = model(X_train)
        Y_train = [Ym_train[:, i] for i in range(p.n_var)]
    # Shapes are (n_train, n_dim)
    dY_dX_train = [tape1.gradient(Y, X_train) for Y in Y_train]
    for i in range(p.n_var):
        np.savetxt(os.path.join(output_dir, "%s_train.dat" %
                   p.dependent_variable_names[i]),
                   tf.reshape(Y_train[i], (n_train,)))
        np.savetxt(os.path.join(output_dir, "del_%s_train.dat" %
                   p.dependent_variable_names[i]), dY_dX_train[i])

    # Save the trained models.
    if save_model:
        model.save(os.path.join(output_dir, "model_" + p.__name__))


if __name__ == "__main__":
    """Begin main program."""
    main()
