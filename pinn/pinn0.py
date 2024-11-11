#!/usr/bin/env python

"""Use neural networks to approximate multivariable scalar functions.

Use neural networks to approximate multivariable scalar functions.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard Python modules.
import copy
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

# Default values for command-line arguments.
DEFAULT_ARGUMENTS = copy.deepcopy(common.DEFAULT_ARGUMENTS)
DEFAULT_ARGUMENTS["data_path"] = None


def create_command_line_parser(description: str = DESCRIPTION):
    """Create the command-line parser.

    Create the command-line parser.

    Parameters
    ----------
    description : str, default DESCRIPTION
        Parser for the command-line.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser for command-line arguments.
    """
    parser = common.create_neural_network_command_line_parser(DESCRIPTION)
    parser.add_argument(
        "data_path",
        help="Path to problem data file"
    )
    return parser


def create_output_directory(clobber: bool = False):
    """Create the output directory for this problem.

    Create the output directory for this problem. The name of the output
    directory is "pinn0".

    Parameters
    ----------
    clobber : bool, default False
        True to delete existing directory of same name.

    Returns
    -------
    output_dir : str
        Path to output directory.

    Raises
    ------
    None
    """
    output_dir = "pinn0"
    if os.path.isdir(output_dir) and clobber:
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    return output_dir


def pinn0(args: dict):
    """Approximate one or more scalar functions with 0-order PINNs.

    Approximate one or more scalar functions with 0-order PINNs.

    Regarding variable names:

    X*: Contains values of independent variables.
    Y*: Contains values of dependent variables.
    XY*: Contains values of independent and dependent variables.

    Parameters
    ----------
    args : dict
        Dictionary of command-line and other options.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Use defaults for unspecified arguments. Merge in additional settings
    # which are passed in by the caller.
    local_args = copy.deepcopy(DEFAULT_ARGUMENTS)
    if args is not None:
        local_args.update(args)
    args = local_args
    if args["debug"]:
        print(f"args = {args}")

    # Local convenience variables
    activation = args["activation"]
    batch_size = args["batch_size"]
    clobber = args["clobber"]
    debug = args["debug"]
    learning_rate = args["learning_rate"]
    load_model = args["load_model"]
    max_epochs = args["max_epochs"]
    n_hid = args["n_hid"]
    n_layers = args["n_layers"]
    nogpu = args["nogpu"]
    precision = args["precision"]
    randomize = args["randomize"]
    save_model = args["save_model"]
    seed = args["seed"]
    verbose = args["verbose"]
    problem_path = args["problem_path"]
    data_path = args["data_path"]

    # ------------------------------------------------------------------------

    # Configure TensorFlow.
    if verbose:
        print("Configuring TensorFlow.")
    common.configure_tensorflow(nogpu, precision, seed)

    # ------------------------------------------------------------------------

    # Import the problem to solve.
    if verbose:
        print(f"Importing python module for problem {problem_path}.")
    p = common.import_problem(problem_path)
    if debug:
        print(f"p = {p}")

    # Create the output directory under the current directory.
    if verbose:
        print("Creating output directory.")
    output_dir = create_output_directory(clobber)
    if debug:
        print(f"output_dir = {output_dir}")

    # Record system information and program arguments.
    if verbose:
        print("Saving system information and program arguments.")
    common.save_system_information(output_dir)
    common.save_arguments(args, output_dir)

    # Save the problem definition and data points.
    if verbose:
        print("Saving problem definition and data points.")
    shutil.copy(problem_path, output_dir)
    shutil.copy(data_path, output_dir)

    # Save a copy of the data points under a standard name.
    path = os.path.join(output_dir, "XY_data.dat")
    shutil.copy(data_path, path)

    # ------------------------------------------------------------------------

    # Load the data points, which specify the function values at each point.
    if verbose:
        print(f"Loading data from {data_path}.")
    XY_data = common.load_problem_data(data_path, precision)
    if debug:
        print(f"XY_data = {XY_data}")

    # Get the count of data points.
    n_data = XY_data.shape[0]
    if debug:
        print(f"n_data = {n_data}")

    # Optionally randomize the order of the training data in-place.
    if randomize:
        if verbose:
            print("Shuffling training data.")
        np.random.shuffle(XY_data)

    # Split the training data into separate arrays for independent and
    # dependent variables.
    X_data = XY_data[:, :p.n_dim]
    Y_data = XY_data[:, p.n_dim:]
    if debug:
        print(f"X_data = {X_data}")
        print(f"Y_data = {Y_data}")

    # ------------------------------------------------------------------------

    # Load or create PINN models for the variables.
    if load_model is not None:
        # <TODO> TEST THIS.
        models = common.load_models(p.dependent_variable_names, load_model)
        # </TODO>
    else:
        models = common.create_models(p.dependent_variable_names, n_layers,
                                      n_hid, activation)
    if debug:
        print(f"models = {models}")

    # ------------------------------------------------------------------------

    # Create the optimizer to use for training.
    if verbose:
        print("Creating optimizer.")
    optimizer = common.create_optimizer(learning_rate)
    if debug:
        print(f"optimizer = {optimizer}")

    # ------------------------------------------------------------------------

    # Prepare inputs for TensorFlow.

    # Convert independent and dependent variables to tf.Variable.
    # Xd, Yd = X and Y for data points
    Xd = tf.Variable(X_data)
    Yd = tf.Variable(Y_data)
    if debug:
        print(f"Xd = {Xd}")
        print(f"Yd = {Yd}")

    # Batch the training points as tf.Variable.
    # Xdbs = X values for data batches
    # Xdbs is a list of tf.Variable.
    # Each tf.Variable has shape (<= batch_size, p.n_dim)
    # Ydbs = Y values for data batches
    # Ydbs is a list of tf.Variable.
    # Each tf.Variable has shape (<= batch_size, p.n_var)
    if verbose:
        print("Batching training data.")
    Xdbs = common.create_tf_batches(X_data, batch_size)
    Ydbs = common.create_tf_batches(Y_data, batch_size)
    n_batches = len(Xdbs)
    if debug:
        print(f"Xdbs = {Xdbs}")
        print(f"Ydbs = {Ydbs}")
        print(f"n_batches = {n_batches}")

    # ------------------------------------------------------------------------

    # Create loss histories by epoch and model (and total).
    losses = np.zeros((max_epochs, p.n_var + 1))

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

        # Part 1: Process each batch of training points.
        for i_batch in range(n_batches):
            if debug:
                print(f"Starting epoch {epoch}, batch {i_batch}.")

            # Xdb is a tf.Variable containing the independent variable values
            # for this batch. It has shape (<=batch_size, p.n_dim).
            # Ydb is a tf.Variable containing the dependent variable values
            # for this batch. It has shape (<=batch_size, p.n_var).
            Xdb = Xdbs[i_batch]
            Ydb = Ydbs[i_batch]
            if debug:
                print(f"Xdb = {Xdb}")
                print(f"Ydb = {Ydb}")

            # Run the forward pass of each model for this batch.
            # tape0 is for computing gradients wrt model parameters.
            with tf.GradientTape(persistent=True) as tape0:

                # Compute the model outputs at the training points.
                # Ymb is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (<= batch_size, 1).
                Ymb = [model(Xdb) for model in models]
                if debug:
                    print(f"Ymb = {Ymb}")

                # Compute the errors in the model-predicted values at each
                # training point for this batch.
                # Ebs is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list.
                # Each Tensor has shape (<= n_train, 1).
                # tf.reshape(Yd[:, i], (Yd.shape[0], 1)
                Ebs = [Ymb[i] - tf.reshape(Ydb[:, i], (Ydb.shape[0], 1))
                       for i in range(p.n_var)]
                if debug:
                    print(f"Ebs = {Ebs}")

                # Compute the loss functions for each model for this batch.
                # The loss function is the RMS error.
                # Lbs is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape () (scalar).
                Lbs = [tf.math.sqrt(tf.reduce_sum(E**2)/E.shape[0])
                       for E in Ebs]
                if debug:
                    print(f"Lbs = {Lbs}")

                # Compute the aggregate loss function for all models for the
                # batch.
                # Tensor has shape () (scalar).
                Lb = tf.reduce_sum(Lbs)
                if debug:
                    print(f"epoch = {epoch}, batch = {i_batch}, Lb = {Lb:.6E}")

                # End of tape0 context.

            # Compute the gradient of the batch loss wrt the model parameters.
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

            # Update the parameters for each model for this batch.
            for (g, m) in zip(pgrad, models):
                optimizer.apply_gradients(zip(g, m.trainable_variables))

            if debug:
                print(f"Ending epoch {epoch}, batch {i_batch}.")

            # End of loop over batches.

        # --------------------------------------------------------------------

        # Step 2: Compute the end-of-epoch loss.

        # Assumes models can process entire training set as a unit.
        sum_E2 = np.zeros(p.n_var)
        for ib in range(n_batches):
            Xdb = Xdbs[ib]
            Ydb = Ydbs[ib]
            Ymbs = [model(Xdb) for model in models]
            Embs = [
                Ymbs[i] - tf.reshape(Ydb[:, i], (Ydb.shape[0], 1))
                for i in range(p.n_var)
            ]
            sum_E2s = np.array([tf.reduce_sum(E**2) for E in Embs])
            sum_E2 += sum_E2s
            # End of data batches.
        Ls = np.array([np.sqrt(E2/n_data) for E2 in sum_E2])
        L = np.sum(Ls)

        # Record the losses for the epoch.
        losses[epoch, :-1] = Ls
        losses[epoch, -1] = L

        # Save the trained models.
        if save_model > 0 and epoch % save_model == 0:
            common.save_models(
                models, p.dependent_variable_names, epoch, output_dir
            )

        if verbose:
            print(f"Epoch = {epoch}, L = {L:E}")

        if debug:
            print(f"Ending epoch {epoch}.")

    # End of training loop.

    # Count the last epoch.
    n_epochs = epoch + 1

    # Print short training summary.
    t_stop = datetime.datetime.now()
    t_elapsed = t_stop - t_start
    if verbose:
        print(f"Training stopped at {t_stop}.")
        print(f"Total training time: {t_elapsed.total_seconds()} seconds")
        print(f"Epochs: {n_epochs}")
        print(f"Final value of loss function: {L}")

    # ------------------------------------------------------------------------

    # Save the final trained models and descriptions.
    if save_model != 0:
        common.save_models(
            models, p.dependent_variable_names, epoch, output_dir
        )

    # Save the loss histories.
    path = os.path.join(output_dir, "L.dat")
    np.savetxt(path, losses)

    # ------------------------------------------------------------------------

    # Return normally.
    return 0


def main():
    """Top-level code for the command-line version of pinn0.

    This is the top-level code for the command-line version of pinn0.
    It processes command-line options, then calls the pinn0() function.

    Parameters
    ----------
    None

    Returns
    -------
    0 on success

    Raises
    ------
    None
    """
    # Set up the command-line parser.
    parser = create_command_line_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}")

    # Convert the arguments from Namespace to dict.
    args = vars(args)

    # Call the main program logic.
    return_code = pinn0(args)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
