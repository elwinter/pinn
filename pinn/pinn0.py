#!/usr/bin/env python

"""Use neural networks to approximate multivariable scalar functions.

Use neural networks to approximate multivariable scalar functions.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard Python modules.
import datetime
import os
import shutil
# import sys

# Import supplemental modules.
import numpy as np
import tensorflow as tf

# Import project modules.
from pinn import common


# Program constants

# Program description.
DESCRIPTION = 'Use a neural network to approximate a function.'


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
    # Create the standard neural network command line parser.
    parser = common.create_neural_network_command_line_argument_parser()
    return parser


def main():
    """Begin main program."""
    # Set up the command-line parser.
    parser = common.create_neural_network_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}")
    activation = args.activation
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
    problem_path = args.problem_path
    training_path = args.training_path

    # ------------------------------------------------------------------------

    # Configure TensorFlow.

    # Set the random number seed for reproducibility.
    if verbose:
        print(f"Seeding Tensorflow random number generator with {seed}.")
    tf.random.set_seed(seed)

    # If requested, disable TensorFlow use of GPU.
    if nogpu:
        if verbose:
            print('Disabling TensorFlow use of GPU.')
        common.disable_gpus()

    # Set the backend TensorFlow precision.
    if verbose:
        print(f"Setting TensorFlow precision to {precision}.")
    tf.keras.backend.set_floatx(precision)

    # ------------------------------------------------------------------------

    # Import the problem to solve.
    if verbose:
        print(f"Importing module for problem {problem_path}.")
    p = common.import_problem(problem_path)
    if debug:
        print(f"p = {p}")

    # Set up the output directory under the current directory.
    # An exception is raised if the directory already exists.
    output_dir = os.path.join('.', f"{p.__name__}-pinn0")
    if debug:
        print(f"output_dir = {output_dir}")
    os.mkdir(output_dir)

    # Record system information, program arguments, problem definition,
    # and training data.
    if verbose:
        print('Recording system information, model hyperparameters, '
              f"problem definition, and training data in {output_dir}.")
    common.save_system_information(output_dir)
    common.save_arguments(args, output_dir)
    shutil.copy(problem_path, output_dir)
    shutil.copy(training_path, output_dir)

    # ------------------------------------------------------------------------

    # Set up Tensorflow data objects.

    # Build one model for each dependent variable in the problem.
    if verbose:
        print('Creating neural network models.')
    models = []
    for i in range(p.n_var):
        if verbose:
            print(f"Creating model for {p.dependent_variable_names[i]}.")
        model = common.build_model(n_layers, H, activation)
        models.append(model)
    if debug:
        print(f"models = {models}")

    # Create the Adam optimizer.
    if verbose:
        print('Creating Adam optimizer.')
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    if debug:
        print(f"optimizer = {optimizer}")

    # ------------------------------------------------------------------------

    # Load the training data.
    XY_train = np.loadtxt(training_path)
    if debug:
        print(f"XY_train = {XY_train}")

    # Count the training points.
    n_train = XY_train.shape[0]
    if debug:
        print(f"n_train = {n_train}")

    # Convert training point locations to tf.Variable.
    X_train = tf.Variable(XY_train[:, :p.n_var], dtype=precision)
    if debug:
        print(f"X_train = {X_train}")

    # Convert training point values to tf.Variable.
    Y_train = tf.Variable(XY_train[:, p.n_var:], dtype=precision)
    if debug:
        print(f"Y_train = {Y_train}")

    # ------------------------------------------------------------------------

    # Create loss history arrays.
    loss = {}
    for v in p.dependent_variable_names:
        loss[v] = {}
        loss[v]['total'] = np.zeros(max_epochs)
    loss['aggregate'] = {}
    loss['aggregate']['total'] = np.zeros(max_epochs)

    # ------------------------------------------------------------------------

    # Train the network models.

    # Record the training start time.
    t_start = datetime.datetime.now()
    if verbose:
        print(f"Training started at {t_start}.")

    # Train for the maximum number of epochs.
    for epoch in range(max_epochs):
        if debug:
            print(f"Starting epoch {epoch}.")

        # Run the forward pass for the training points in a single batch.
        # tape0 is for computing gradients wrt network parameters.
        with tf.GradientTape(persistent=True) as tape0:

            # Compute the network outputs at the training points.
            # Y_model is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape (n_train, 1).
            Y_model = [model(X_train) for model in models]
            if debug:
                print(f"Y_model = {Y_model}")

            # Compute the errors in the models at each training point.
            # E_model is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_train, 1).
            E_model = [
                Y_model[i] - tf.reshape(Y_train[:, i], (n_train, 1))
                for i in range(p.n_var)
            ]
            if debug:
                print(f"E_model = {E_model}")

            # Compute and save the individual loss functions for each model.
            # The loss function is the RMS error.
            # L_model is a list of Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape () (scalar).
            L_model = [
                tf.math.sqrt(tf.reduce_sum(E**2)/E.shape[0]) for E in E_model
            ]
            if debug:
                print(f"L_model = {L_model}")
            for i in range(p.n_var):
                varname = p.dependent_variable_names[i]
                loss[varname]['total'][epoch] = L_model[i].numpy()

            # Compute and save the aggregate loss function.
            # Tensor has shape () (scalar).
            L = tf.reduce_sum(L_model)
            if verbose:
                print(f"epoch = {epoch}, L = {L:.6E}")
            loss['aggregate']['total'][epoch] = L.numpy()

        # Compute the gradient of the loss wrt the network parameters.
        # pgrad is a list of lists of Tensor objects.
        # There are p.n_var sub-lists in the top-level list (one per
        # model).
        # There are 3 Tensors in each sub-list, with shapes based on
        # model.trainable_variables.
        pgrad = [
            tape0.gradient(L, model.trainable_variables) for model in models
        ]
        if debug:
            print(f"pgrad = {pgrad}")

        # Update the parameters for this epoch.
        for (g, m) in zip(pgrad, models):
            optimizer.apply_gradients(zip(g, m.trainable_variables))

        # Save the trained models.
        if save_model > 0 and epoch % save_model == 0:
            for (i, model) in enumerate(models):
                path = os.path.join(
                    output_dir, 'models', f"{epoch}",
                    f"model_{p.dependent_variable_names[i]}"
                )
                model.save(path)

        if debug:
            print(f"Ending epoch {epoch}.")

    # End of training loop.

    # Record the training end time.
    t_stop = datetime.datetime.now()
    if verbose:
        print(f"Training stopped at {t_stop}.")

    # Determine actual number of epochs used in case training loop ended
    # early.
    n_epochs = epoch + 1

    # Print short training summary.
    t_elapsed = t_stop - t_start
    if verbose:
        print(f"Total training time: {t_elapsed.total_seconds()} seconds")
        print(f"Epochs: {n_epochs}")
        print(f"Final value of loss function: {L}")

    # Save the final trained models.
    if save_model != 0:
        for (i, model) in enumerate(models):
            path = os.path.join(
                output_dir, 'models', f"{epoch}",
                f"model_{p.dependent_variable_names[i]}"
            )
            model.save(path)

    # Save the loss histories.
    for v in p.dependent_variable_names:
        path = os.path.join(output_dir, f"L_{v}.dat")
        np.savetxt(path, loss[v]['total'])
    path = os.path.join(output_dir, 'L.dat')
    np.savetxt(path, loss['aggregate']['total'])


if __name__ == '__main__':
    """Begin main program."""
    main()
