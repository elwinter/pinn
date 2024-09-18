#!/usr/bin/env python

"""Use neural networks to approximate mathematical functions.

Use neural networks to approximate mathematical functions.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard Python modules.
import datetime
import os
import shutil
import sys

# Import supplemental modules.
import numpy as np
import tensorflow as tf

# Import project modules.
from pinn import common


# Program constants

# Program description.
DESCRIPTION = "Use a neural network to approximate a function."


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
    # Create the standard argument parser for neural network code.
    parser = common.create_neural_network_command_line_argument_parser(
        DESCRIPTION
    )

    # Add arguments specific to this script.
    parser.add_argument(
        "--randomize", action="store_true",
        help="Randomize order of training data (default: %(default)s)"
    )
    parser.add_argument(
        "problem_path",
        help="Path to problem description file (in python)"
    )
    parser.add_argument(
        "data_path",
        help="Path to file of training data"
    )

    # Return the parser.
    return parser


def pinn0(args: dict):
    """Primary entry point for PINN function approximation code.

    Use a PINN to approximate a mathematical function.

    Regarding variable names:

    X*: Contains values of independent variables.
    Y*: Contains values of dependent variables.
    XY*: Contains values of independent and dependent variables.
    *b*: Quantity is for a batch.
    *m*: Quantity of for a model.

    Parameters
    ----------
    args: dict
        Dictionary of command-line options.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Local convenience variables
    batch_size = args.get("batch_size", -1)
    clobber = args.get("clobber", False)
    debug = args.get("debug", False)
    learning_rate = args.get("learning_rate", 0.01)
    load_models = args.get("load_models", None)
    max_epochs = args.get("max_epochs", 0)
    precision = args.get("precision", "float32")
    randomize = args.get("randomize", False)
    save_model = args.get("save_model", -1)
    verbose = args.get("verbose", False)
    problem_path = args.get("problem_path", None)
    data_path = args.get("data_path", None)

    # Basic sanity checks.
    assert batch_size == -1 or batch_size > 0
    assert max_epochs > 0
    assert problem_path is not None
    assert data_path is not None

    # ------------------------------------------------------------------------

    # Configure TensorFlow.
    if verbose:
        print("Configuring TensorFlow.")
    common.configure_tensorflow(args)

    # ------------------------------------------------------------------------

    # Import the problem to solve.
    if verbose:
        print(f"Importing problem {problem_path}.")
    p = common.import_problem(problem_path)
    if debug:
        print(f"p = {p}")

    # Create the output directory under the current directory.
    if verbose:
        print("Creating output directory.")
    output_dir = common.create_output_directory(p, "-pinn0", clobber=clobber)
    if debug:
        print(f"output_dir = {output_dir}")

    # Record system information and program arguments.
    if verbose:
        print("Saving system information and program arguments.")
    common.save_system_information(output_dir)
    common.save_arguments(args, output_dir)

    # Copy the problem definition and data points.
    if verbose:
        print("Copying problem definition and data points.")
    shutil.copy(problem_path, output_dir)
    shutil.copy(data_path, output_dir)

    # Save a copy of the data points under a standard name.
    if data_path.endswith(".gz"):
        path = os.path.join(output_dir, "XY_data.dat.gz")
    else:
        path = os.path.join(output_dir, "XY_data.dat")
    shutil.copy(data_path, path)

    # ------------------------------------------------------------------------

    # Load the data points, which specify the function values at each point.
    # The data is always returned as a 2-D numpy array of shape
    # (n_data, p.n_dim + p.n_var).
    if verbose:
        print(f"Reading problem data from {data_path}.")
    XY_data = common.load_problem_data(data_path, precision=precision)
    if debug:
        print(f"XY_data = {XY_data}")

    # Get the count of data points.
    n_data = XY_data.shape[0]
    if debug:
        print(f"n_data = {n_data}")

    # Extract the *locations* of the data points.
    # Shape is (n_data, p.n_dim)
    X_data = XY_data[:, :p.n_dim]
    if debug:
        print(f"X_data = {X_data}")

    # Extract the *values* of the data points.
    # Shape is (n_data, p.n_var)
    Y_data = XY_data[:, p.n_dim:]
    if debug:
        print(f"Y_data = {Y_data}")

    # ------------------------------------------------------------------------

    # Load or create PINN models for the variables.
    if load_models:
        if verbose:
            print(f"Loading trainied models from {load_models}.")
        models = common.load_trained_models(
            load_models, p.dependent_variable_names
        )
    else:
        if verbose:
            print("Creating untrained models.")
        models = common.create_models(p, args)
    if debug:
        print(f"models = {models}")

    # ------------------------------------------------------------------------

    # Create the optimizer to use for training.
    if verbose:
        print("Creating optimizer for training.")
    optimizer = common.create_optimizer(learning_rate)
    if debug:
        print(f"optimizer = {optimizer}")

    # ------------------------------------------------------------------------

    # Randomize the data if needed.
    if randomize:
        if verbose:
            print("Randomizing order of training data.")
        np.random.shuffle(XY_data)

    # Split the data into batches of TensorFlow Variables.
    if verbose:
        print("Batching training data.")
    batches = common.create_batches(XY_data, batch_size)
    if debug:
        print(f"batches = {batches}")

    # ------------------------------------------------------------------------

    # Create loss histories by epoch and model as Python lists, so they can be
    # easily updated. Shape is (max_epochs, n_batches, p.n_var + 1), where
    # there is one plane per epoch, one row per batch, and one column per
    # dependent variable, with an extra column for the aggregate loss.
    loss = np.zeros((max_epochs, len(batches), p.n_var + 1))
    Le = np.zeros((max_epochs,))

    # ------------------------------------------------------------------------

    # Train the models.

    # Record the training start time.
    t_start = datetime.datetime.now()
    if verbose:
        print(f"Training started at {t_start}.")

    # Train for the maximum number of epochs.
    for epoch in range(max_epochs):
        if debug:
            print(f"Starting epoch {epoch}.")

        # Train using each batch for this epoch.
        for (ib, XYb) in enumerate(batches):
            if debug:
                print(f"Starting batch {ib}.")

            # Run the forward pass for this batch.
            with tf.GradientTape(persistent=True) as tape0:

                # Compute the model outputs at the training points.
                # Ybm is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (batch_size, 1).
                shape = (XYb.shape[0], p.n_dim)
                Xb = tf.reshape(XYb[:, :p.n_dim], shape=shape)
                Ybm = [model(Xb) for model in models]
                if debug:
                    print(f"Ybm = {Ybm}")

                # Compute the errors in the models at each training point.
                # Eb is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list.
                # Each Tensor has shape (batch_size, 1).
                Eb = [0.0]*p.n_var
                for iv in range(p.n_var):
                    Ybt = tf.reshape(XYb[:, p.n_dim + iv], (XYb.shape[0], 1))
                    Eb[iv] = Ybm[iv] - Ybt
                if debug:
                    print(f"Eb = {Eb}")

                # Compute the loss function for each model.
                # The loss function is the RMS error.
                # Lbm is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape () (scalar).
                Lbm = [
                    tf.math.sqrt(tf.reduce_sum(E**2)/E.shape[0]) for E in Eb
                ]
                if debug:
                    print(f"Lbm = {Lbm}")

                # Compute the total loss function for the batch.
                # Tensor has shape () (scalar).
                Lb = tf.reduce_sum(Lbm)
                if debug:
                    print(f"Lb = {Lb}")

                # Save the loss history.
                for im in range(len(models)):
                    loss[epoch][ib][im] = Lbm[im].numpy()
                loss[epoch][ib][-1] = Lb

                # End of tape0 context.

            # Compute the gradient of the loss wrt network parameters.
            # pgrad is a list of lists of Tensor objects.
            # There are p.n_var sub-lists in the top-level list (one per
            # model).
            # There are 3 Tensors in each sub-list, with shapes based on
            # model.trainable_variables.
            pgrad = [
                tape0.gradient(Lb, model.trainable_variables)
                for model in models
            ]
            if debug:
                print(f"pgrad = {pgrad}")

            # Update the parameters for this epoch and batch.
            for (g, m) in zip(pgrad, models):
                optimizer.apply_gradients(zip(g, m.trainable_variables))

            if debug:
                print(f"Ending batch {ib}.")

        # Compute the predicted values at the end of the epoch.
        Yem = [model(X_data) for model in models]
        if debug:
            print(f"Yem = {Yem}")

        # Compute the errors at the end of the epoch.
        Ee = [0.0]*p.n_var
        for iv in range(p.n_var):
            Yet = tf.reshape(XY_data[:, p.n_dim + iv], (XY_data.shape[0], 1))
            Ee[iv] = Yem[iv] - Yet
        if debug:
            print(f"Ee = {Ee}")

        # Compute the losses at the end of the epoch.
        Lem = [
            tf.math.sqrt(tf.reduce_sum(E**2)/E.shape[0]) for E in Ee
        ]
        if debug:
            print(f"Lem = {Lem}")
        Le[epoch] = tf.reduce_sum(Lem)
        if debug:
            print(f"Le = {Le}")

        if verbose:
            print(f"Epoch = {epoch:6d}, Le = {Le[epoch]:.6E}")

        # Save the trained models.
        if save_model > 0 and epoch % save_model == 0:
            for (im, model) in enumerate(models):
                path = os.path.join(
                    output_dir, "models", f"{epoch:06d}",
                    p.dependent_variable_names[im]
                )
                model.save(path)

        if debug:
            print(f"Ending epoch {epoch}.")

    # End of training loop.

    # Record the training end time.
    t_stop = datetime.datetime.now()
    if verbose:
        print(f"Training stopped at {t_stop}.")

    # Count the last epoch.
    n_epochs = epoch + 1
    if debug:
        print(f"n_epochs = {n_epochs}")

    # Print short training summary.
    t_elapsed = t_stop - t_start
    if verbose:
        print(f"Total training time: {t_elapsed.total_seconds()} seconds")
        print(f"Epochs: {n_epochs}")
        print(f"Final value of loss function: {loss[-1][-1][-1]}")

    # Save the final trained models and descriptions.
    if save_model != 0:
        for (im, model) in enumerate(models):

            # Save the trained model.
            path = os.path.join(
                output_dir, "models", f"{epoch:06d}",
                p.dependent_variable_names[im]
            )
            model.save(path)

            # Save a text description of the model.
            variable_name = p.dependent_variable_names[im]
            path = os.path.join(output_dir, "models",
                                f"model_{variable_name}.txt")
            old_stdout = sys.stdout
            with open(path, "w", encoding="utf-8") as f:
                sys.stdout = f
                model.summary()
            sys.stdout = old_stdout

    # Save the loss histories as a binary NumPy file.
    path = os.path.join(output_dir, "loss")
    np.save(path, loss)
    path = os.path.join(output_dir, "Le.dat")
    np.savetxt(path, Le)


def main():
    """Driver for command-line version of code."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}")

    # Pass the command-line arguments to the main function as a dict.
    args = vars(args)
    pinn0(args)


if __name__ == "__main__":
    main()
