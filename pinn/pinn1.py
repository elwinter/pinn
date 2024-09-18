#!/usr/bin/env python

"""Use PINNs to solve a set of coupled 1st-order PDE.

This program will use a set of Physics-Informed Neural Networks (PINNs) to
solve a set of coupled 1st-order PDEs.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard Python modules.
import datetime
import os
import shutil
import sys

# Import 3rd-party modules.
import numpy as np
import tensorflow as tf

# Import project modules.
from pinn import common


# Program constants

# Program description
DESCRIPTION = "Solve a set of coupled 1st-order PDE using the PINN method."


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
        "--w_data", "-w", type=float, default=0.0,
        help="Normalized weight for data loss function "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "problem_path",
        help="Path to problem description file (in python)"
    )
    parser.add_argument(
        "data_path",
        help="Path to problem data (IC, BC, etc.) file"
    )
    parser.add_argument(
        "training_path",
        help="Path to training points file"
    )
    return parser


def pinn1(args: dict):
    """Primary entry point for PINN 1st-order solution code.

    Use a PINN to solve a set of 1st-order differential equations.

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
    # Parse the command-line arguments.
    activation = args.get("activation", "sigmoid")
    batch_size = args.get("batch_size", -1)
    clobber = args.get("clobber", False)
    debug = args.get("debug", False)
    learning_rate = args.get("learning_rate", 0.01)
    load_model = args.get("load_model", None)
    max_epochs = args.get("max_epochs", 0)
    H = args.get("n_hid", 10)
    n_layers = args.get("n_layers", 1)
    precision = args.get("precision", "float32")
    randomize = args.get("randomize", False)
    save_model = args.get("save_model", -1)
    verbose = args.get("verbose", False)
    w_data = args.get("w_data", 0.0)
    problem_path = args.get("problem_path", None)
    data_path = args.get("data_path", None)
    training_path = args.get("training_path", None)

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
    output_dir = common.create_output_directory(p, "-pinn1", clobber=clobber)
    if debug:
        print(f"output_dir = {output_dir}")

    # Record system information and program arguments.
    if verbose:
        print("Saving system information and program arguments.")
    common.save_system_information(output_dir)
    common.save_arguments(args, output_dir)

    # Copy the problem definition, data, and training points.
    if verbose:
        print("Copying problem definition, data, and training points.")
    shutil.copy(problem_path, output_dir)
    shutil.copy(data_path, output_dir)
    shutil.copy(training_path, output_dir)

    # Save a copy of the data and training points under standard names.
    if data_path.endswith(".gz"):
        path = os.path.join(output_dir, "XY_data.dat.gz")
    else:
        path = os.path.join(output_dir, "XY_data.dat")
    shutil.copy(data_path, path)
    if training_path.endswith(".gz"):
        path = os.path.join(output_dir, "X_train.dat.gz")
    else:
        path = os.path.join(output_dir, "X_train.dat")
    shutil.copy(training_path, path)

    # ------------------------------------------------------------------------

    # Load the training points. These are just coordinate tuples, one per
    # line, space-delimited.
    if verbose:
        print(f"Reading training points from {training_path}.")
    X_train = common.load_training_data(training_path, precision=precision)
    # X_train is np.ndarray of shape (n_train, p.n_dim) OR (n_train,) for 1D.
    # X_train = np.loadtxt(training_path, dtype=precision)
    if debug:
        print(f"X_train = {X_train}")

    # Count the training points.
    n_train = X_train.shape[0]
    if debug:
        print(f"n_train = {n_train}")

    # -------------------------------------------------------------------------

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

    # Extract the *locations* of the supplied data points.
    # Shape is (n_data, p.n_dim)
    X_data = XY_data[:, :p.n_dim]
    if debug:
        print(f"X_data = {X_data}")

    # Extract the *values* of the supplied data points.
    # Shape is (n_data, p.n_var)
    Y_data = XY_data[:, p.n_dim:]
    if debug:
        print(f"Y_data = {Y_data}")

    # ------------------------------------------------------------------------

    # Load or create PINN models for the variables.
    if load_model:
        if verbose:
            print(f"Loading trainied models from {load_model}.")
        models = common.load_trained_models(
            load_model, p.dependent_variable_names
        )
    else:
        if verbose:
            print("Creating untrained models.")
        models = common.create_models(p, args)
    if debug:
        print(f"models = {models}")

    # -------------------------------------------------------------------------

    # Create the optimizer to use for training.
    if verbose:
        print("Creating optimizer for training.")
    optimizer = common.create_optimizer(learning_rate)
    if debug:
        print(f"optimizer = {optimizer}")

    # -------------------------------------------------------------------------

    # Randomize the data if needed.
    if randomize:
        if verbose:
            print("Randomizing order of training data.")
        np.random.shuffle(XY_data)

    # Split the training points into batches of TensorFlow Variables.
    if verbose:
        print("Batching training points.")
    batches = common.create_batches(X_train, batch_size)
    if debug:
        print(f"batches = {batches}")

    # -------------------------------------------------------------------------

    # Create loss histories by epoch, batch, and model, so they can be
    # easily updated. Shape is (max_epochs, n_batches, p.n_var + 1), where
    # there is one plane per epoch, one row per batch, and one column per
    # dependent variable, with an extra column for the aggregate loss.
    _loss = np.zeros((max_epochs, len(batches), p.n_var + 1, 3))
    _Le = np.zeros((max_epochs, 3))

    # ------------------------------------------------------------------------

    # Compute weights for residual and data loss functions.
    w_res = 1.0 - w_data
    if debug:
        print(f"w_res = {w_res}")
        print(f"w_data = {w_data}")

    # -------------------------------------------------------------------------

    # Prepare inputs for TensorFlow.

    # Convert training points to tf.Variable.
    if verbose:
        print("Converting training points to TensorFlow Variable")
    X_train_tf = tf.Variable(X_train)
    if debug:
        print(f"X_train_tf = {X_train_tf}")

    # Convert data locations to tf.Variable.
    if verbose:
        print("Converting data locations to TensorFlow Variable")
    X_data_tf = tf.Variable(X_data)
    if debug:
        print(f"X_data_tf = {X_data_tf}")

    # Convert data values to tf.Variable.
    if verbose:
        print("Converting data values to TensorFlow Variable")
    Y_data_tf = tf.Variable(Y_data)
    if debug:
        print(f"Y_data_tf = {Y_data_tf}")

    # -------------------------------------------------------------------------

    # Create loss histories as Python lists, so they can be easily updated.
    loss = {}
    for v in p.dependent_variable_names:
        loss[v] = {}
        loss[v]["residual"] = []
        loss[v]["data"] = []
        loss[v]["total"] = []
    loss["aggregate"] = {}
    loss["aggregate"]["residual"] = []
    loss["aggregate"]["data"] = []
    loss["aggregate"]["total"] = []

    # -------------------------------------------------------------------------

    # Train the models.

    # Training involves presenting the training points and data points
    # together, a total of max_epochs times.

    # Record the training start time.
    t_start = datetime.datetime.now()
    if verbose:
        print(f"Training started at {t_start}.")

    # Main training loop
    for epoch in range(max_epochs):
        if debug:
            print(f"Starting epoch {epoch}.")

        # --------------------------------------------------------------------

        # _train : computed using training points
        # _data : computed using data points
        # _model : computed using model

        # Run the forward pass for this epoch.
        # tape0 is for computing gradients wrt network parameters.
        # tape1 is for computing 1st-order derivatives of outputs wrt
        # inputs.
        with tf.GradientTape(persistent=True) as tape0:
            with tf.GradientTape(persistent=True) as tape1:

                # Compute the network outputs at all training points. These
                # are the values of the dependent variables Y to use in the
                # differential equations G.
                # Y_train_model is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (n_train, 1).
                Y_train_model = []
                Y_train_model = [model(X_train_tf) for model in models]
                if debug:
                    print(f"Y_train_model = {Y_train_model}")

                # Compute the network outputs at all data points. These
                # are the values of the dependent variables Y to use when
                # comparing to the supplied data.
                # Y_data_model is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (n_data, 1).
                Y_data_model = []
                Y_data_model = [model(X_data_tf) for model in models]
                if debug:
                    print(f"Y_data_model = {Y_data_model}")

            # Compute the gradients of the network outputs wrt inputs for
            # the training points. These are the values of the partial
            # derivatives dY/dX to use in the differential equations G.
            # dY_dX_train_model is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape (n_train, p.n_dim).
            dY_dX_train_model = [tape1.gradient(Y, X_train_tf)
                                 for Y in Y_train_model]
            if debug:
                print(f"dY_dX_train_model = {dY_dX_train_model}")

            # Compute the values of the differential equations at all
            # training points.
            # G_train_model is a list of Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape (n_train, 1).
            G_train_model = [f(X_train_tf, Y_train_model, dY_dX_train_model)
                             for f in p.de]
            if debug:
                print(f"G_train_model = {G_train_model}")

            # -----------------------------------------------------------------

            # Compute the loss function for the equation residuals at the
            # training points for each model.
            # L_res_per_model is a list of Tensor objects.
            # There are p.n_var Tensors in the list (one per equation).
            # Each Tensor has shape () (scalar).
            L_res_per_model = [
                tf.math.sqrt(tf.reduce_sum(G**2)/n_train)
                for G in G_train_model
            ]
            if debug:
                print(f"L_res_per_model = {L_res_per_model}")

            # Compute the errors in the predicted values at the data points.
            # E_data_per_model is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_data, 1).
            E_data_per_model = [
                Y_data_model[i] - tf.reshape(Y_data_tf[:, i], (n_data, 1))
                for i in range(p.n_var)
            ]
            if debug:
                print(f"E_data_per_model = {E_data_per_model}")

            # Compute the loss functions for the data points for each
            # model.
            # L_data_per_model is a list of Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape () (scalar).
            L_data_per_model = [
                tf.math.sqrt(tf.reduce_sum(E**2)/n_data) for E in E_data_per_model
            ]
            if debug:
                print(f"L_data_per_model = {L_data_per_model}")

            # Compute the weighted aggregate loss function per model.
            L_per_model = [
                w_res*L1 + w_data*L2 for (L1, L2)
                in zip(L_res_per_model, L_data_per_model)
            ]
            if debug:
                print(f"L_per_model = {L_per_model}")

            # Compute the aggregated residual loss function.
            L_res = tf.math.reduce_sum(L_res_per_model)
            if debug:
                print(f"L_res = {L_res}")

            # Compute the aggregated data loss function.
            L_data = tf.math.reduce_sum(L_data_per_model)
            if debug:
                print(f"L_data = {L_data}")

            # Compute the weighted aggregate loss function.
            L = w_res*L_res + w_data*L_data
            if debug:
                print(f"L = {L}")

            # Save the losses for this epoch.
            for (i, v) in enumerate(p.dependent_variable_names):
                loss[v]["residual"].append(L_res_per_model[i].numpy())
                _loss[epoch][0][i][0] = L_per_model[i].numpy()
                loss[v]["data"].append(L_data_per_model[i].numpy())
                _loss[epoch][0][i][1] = L_per_model[i].numpy()
                loss[v]["total"].append(L_per_model[i].numpy())
                _loss[epoch][0][i][2] = L_per_model[i].numpy()
            loss["aggregate"]["residual"].append(L_res.numpy())
            _Le[epoch][0] = L_res.numpy()
            loss["aggregate"]["data"].append(L_data.numpy())
            loss["aggregate"]["total"].append(L.numpy())
            if debug:
                print(f"loss = {loss}")

        # Compute the gradient of the weighted aggregate loss function wrt
        # the network parameters.
        # pgrad is a list of lists of Tensor objects.
        # There are p.n_var sub-lists in the top-level list (one per
        # model).
        # There are 3 Tensors in each sub-list, with shapes:
        # Input weights: (p.n_dim, H)
        # Input biases: (H,)
        # Output weights: (H, 1)
        # Each Tensor is shaped based on model.trainable_variables.
        pgrad = [
            tape0.gradient(L, model.trainable_variables)
            for model in models
        ]
        if debug:
            print(f"pgrad = {pgrad}")

        # Update the parameters for this epoch.
        for (g, m) in zip(pgrad, models):
            optimizer.apply_gradients(zip(g, m.trainable_variables))

        # --------------------------------------------------------------------

        if verbose:
            print(f"epoch = {epoch}, (L_res, L_data, L) = "
                f"({L_res:6e}, {L_data:6e}, {L:6e})")

        # Save the trained models.
        if save_model > 0 and epoch % save_model == 0:
            for (i, model) in enumerate(models):
                path = os.path.join(
                    output_dir, "models", f"{epoch:06d}",
                    f"model_{p.dependent_variable_names[i]}"
                )
                model.save(path)

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
        print(f"Total training time: {t_elapsed.total_seconds()} seconds",
              flush=True)
        print(f"Epochs: {n_epochs}")
        print(f"Final value of loss function: {L}")

    # Save the final trained models and descriptions.
    if save_model != 0:
        for (i, model) in enumerate(models):
            path = os.path.join(
                output_dir, "models", f"{epoch:06d}",
                f"model_{p.dependent_variable_names[i]}"
            )
            model.save(path)
            variable_name = p.dependent_variable_names[i]
            path = os.path.join(output_dir, "models", f"model_{variable_name}.txt")
            old_stdout = sys.stdout
            with open(path, "w") as f:
                sys.stdout = f
                model.summary()
            sys.stdout = old_stdout

    # Save the loss histories.
    for (i, v) in enumerate(p.dependent_variable_names):
        np.savetxt(
            os.path.join(output_dir, f"L_res_{v}.dat"), loss[v]["residual"]
        )
        np.savetxt(
            os.path.join(output_dir, f"L_data_{v}.dat"), loss[v]["data"]
        )
        np.savetxt(
            os.path.join(output_dir, f"L_{v}.dat"), loss[v]["total"]
        )
    np.savetxt(
        os.path.join(output_dir, "L_res.dat"), loss["aggregate"]["residual"]
    )
    np.savetxt(
        os.path.join(output_dir, "L_data.dat"), loss["aggregate"]["data"]
    )
    np.savetxt(
        os.path.join(output_dir, "L.dat"), loss["aggregate"]["total"]
    )


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
    pinn1(args)


if __name__ == "__main__":
    main()
