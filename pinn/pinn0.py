#!/usr/bin/env python

"""Use neural networks to approximate multivariable scalar functions.

Use neural networks to approximate multivariable scalar functions.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard Python modules.
import argparse
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
    parser = argparse.ArgumentParser(DESCRIPTION)
    parser.add_argument(
        "--activation", "-a", default="sigmoid",
        help="Specify activation function (default: %(default)s)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=-1,
        help="Batch size (-1 for single batch) (default: %(default)s)"
    )
    parser.add_argument(
        "--clobber", action="store_true",
        help="Overwrite existing output directory (default: %(default)s)"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01,
        help="Initial learning rate for training (default: %(default)s)"
    )
    parser.add_argument(
        "--load_model", default=None,
        help="Path to directory containing models to load (default:"
             " %(default)s)"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100,
        help="Maximum number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--multi", action="store_true",
        help="Use a single multi-output network (default: %(default)s)"
    )
    parser.add_argument(
        "--n_hid", type=int, default=10,
        help="Number of hidden nodes per layer (default: %(default)s)"
    )
    parser.add_argument(
        "--n_layers", type=int, default=1,
        help="Number of hidden layers (default: %(default)s)"
    )
    parser.add_argument(
        "--nogpu", action="store_true",
        help="Disable TensorFlow use of GPU(s) (default: %(default)s)"
    )
    parser.add_argument(
        "--precision", type=str, default="float32",
        help="Precision to use in TensorFlow solution (default: %(default)s)"
    )
    parser.add_argument(
        "--save_model", type=int, default=-1,
        help="Save interval (epochs) for trained model (0 = do not save, "
        "-1 = save at end, n > 0 = save every n epochs) (default: %(default)s)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for random number generator (default: %(default)s)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "problem_path",
        help="Path to problem description file (in python)"
    )
    parser.add_argument(
        "data_path",
        help="Path to problem data file"
    )
    return parser


def configure_tensorflow(args: dict):
    """Configure TensorFlow.

    Configure TensorFlow.

    Parameters
    ----------
    args : dict
        Dictionary of command-line options.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    if args["verbose"]:
        print("Configuring TensorFlow.")

    # If requested, disable TensorFlow use of GPU.
    if args["nogpu"]:
        if args["verbose"]:
            print("Disabling TensorFlow use of GPU.")
        common.disable_gpus()

    # Set the backend TensorFlow precision.
    if args["verbose"]:
        print(f"Setting TensorFlow precision to {args['precision']}.")
    tf.keras.backend.set_floatx(args["precision"])

    # Set the random number seed for reproducibility.
    if args["verbose"]:
        print("Seeding TensorFlow random number generator with "
              f"{args['seed']}.")
    tf.random.set_seed(args['seed'])


def create_output_directory(p, args: dict):
    """Create the output directory for this problem.

    Create the output directory for this problem. The name of the output
    directory is the name of the problem python module, with "-pinn0"
    appended to the end of the name.

    Parameters
    ----------
    p : Python module object
        Imported module for the problem definition.
    args : dict
        Dictionary of command-line options.

    Returns
    -------
    output_dir : str
        Name of output directory.

    Raises
    ------
    None
    """
    output_dir = os.path.join(".", f"{p.__name__}-pinn0")
    if args["debug"]:
        print(f"output_dir = {output_dir}")
    if os.path.isdir(output_dir) and args["clobber"]:
        if args["verbose"]:
            print(f"Removing existing output directory {output_dir}.")
        shutil.rmtree(output_dir)
    if args["verbose"]:
        print(f"Creating output directory {output_dir}.")
    os.mkdir(output_dir)
    return output_dir


def load_problem_data(args: dict):
    """Load the problem data.

    Load the problem data, which specifies the function values at each
    point.

    Each line contains a tuple of coordinates (the independent variable values
    for a location), followed by a tuple of dependent variable values,
    containing the values of each dependent variable at that location.

    Parameters
    ----------
    args : dict
        Dictionary of command-line options.

    Returns
    -------
    XY_data : np.ndarray, shape(n_data, p.n_dim + p.n_var)
        Array of data points.

    Raises
    ------
    None
    """
    if args["verbose"]:
        print(f"Reading problem data from {args['data_path']}.")
    XY_data = np.loadtxt(args["data_path"], dtype=args["precision"])
    if args["debug"]:
        print(f"XY_data = {XY_data}")

    # If the problem data shape is 1-D (a single point), reshape to 2-D,
    # (1, n_data) to make compatible with later TensorFlow calls, which
    # expect a 2D Tensor.
    if len(XY_data.shape) == 1:
        if args["verbose"]:
            print("Problem data is 1-D (single point), reshaping to 2-D.")
        XY_data = XY_data.reshape(1, XY_data.shape[0])
        if args["debug"]:
            print(f"Reshaped XY_data = {XY_data}")

    # Return the problem data.
    return XY_data


def load_or_create_models(p, args: dict):
    """Load or create the PINN models.

    Load or create the PINN models.

    Parameters
    ----------
    p : Python module object
        Imported module for the problem definition.
    args : dict
        Dictionary of command-line options.

    Returns
    -------
    models : list of keras.src.engine.sequential.Sequential
        The new models.

    Raises
    ------
    TypeError
        If --load_model is specified.
    """
    models = []
    if args["multi"]:
        if args["load_model"]:
            raise TypeError("No --load_model yet!")
        else:
            if args["verbose"]:
                print("Creating untrained multi-output model.")
            model = common.build_multi_output_model(
                args["n_layers"], args["n_hid"], args["activation"], p.n_var
            )
            if args["debug"]:
                print(f"model = {model}")
            models.append(model)
    else:
        if args["load_model"]:
            raise TypeError("No --load_model yet!")
            # if args["verbose"]:
            #     print(f"Loading trained models from {args['load_model']}.")
            # for (i, v) in enumerate(p.dependent_variable_names):
            #     if verbose:
            #         print(f"Loading model for {v}.")
            #     path = os.path.join(load_model, f"model_{v}")
            #     if debug:
            #         print(f"path = {path}")
            #     model = tf.keras.models.load_model(path)
            #     if debug:
            #         print(f"model = {model}")
            #     models.append(model)
        else:
            if args["verbose"]:
                print("Creating untrained models.")
            for v in p.dependent_variable_names:
                if args["verbose"]:
                    print(f"Creating untrained model for {v}.")
                model = common.build_model(
                    args["n_layers"], args["n_hid"], args["activation"]
                )
                if args["debug"]:
                    print(f"model = {model}")
                models.append(model)
    if args["debug"]:
        print(f"models = {models}")

    # Return the models.
    return models


def create_optimizer(args: dict):
    """Create the training optimizer.

    Create the training optimizer.

    Parameters
    ----------
    args : dict
        Dictionary of command-line options.

    Returns
    -------
    optimizer : keras.src.optimizers.legacy.adam.Adam
        The optimizer to use for training.

    Raises
    ------
    None
    """
    if args["verbose"]:
        print("Creating Adam optimizer.")
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=args["learning_rate"]
    )
    if args["debug"]:
        print(f"optimizer = {optimizer}")

    # Return the optimizer.
    return optimizer


def create_batches(X_train: np.ndarray, args: dict):
    """Split the data into batches of tf.Variable.

    Split the data into batches of tf.Variable.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, p.n_dim)
        Array of training points.
    args : dict
        Dictionary of command-line arguments.

    Returns
    -------
    training_batches : list of tf.Variable
        Training points split into batches.

    Raises
    ------
    None
    """
    if args["verbose"]:
        print("Converting training points to TensorFlow Variable.")
    training_batches = []
    batch_size = args["batch_size"]
    n_train = X_train.shape[0]
    if batch_size == -1:
        if args["verbose"]:
            print("Using single batch for training points.")
        X_train_batch_tf = tf.Variable(X_train)
        training_batches.append(X_train_batch_tf)
    else:
        n_batches = int(np.ceil(n_train/batch_size))
        if args["debug"]:
            print(f"n_batches = {n_batches}")
        for ib in range(n_batches):
            if args["verbose"]:
                print(f"Creating batch {ib}.")
            i_start = ib*batch_size
            i_end = (ib + 1)*batch_size
            i_end = min(i_end, n_train)
            if args["debug"]:
                print(f"i_start, i_end = {i_start}, {i_end}")
            X_train_batch_np = X_train[i_start:i_end]
            if args["debug"]:
                print(f"X_train_batch_np = {X_train_batch_np}")
            X_train_batch_tf = tf.Variable(X_train_batch_np)
            training_batches.append(X_train_batch_tf)
    if args["debug"]:
        print(f"training_batches = {training_batches}")

    # Return the list of batches.
    return training_batches


def pinn0(args: dict):
    """Primary entry point for 0th-order PINN code.

    Use a 0th-order PINN to approximate a function.

    Regarding variable names:

    X*: Contains values of independent variables.
    Y*: Contains values of dependent variables.
    XY*: Contains values of independent and dependent variables.

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
    activation = args["activation"]
    debug = args["debug"]
    learning_rate = args["learning_rate"]
    max_epochs = args["max_epochs"]
    H = args["n_hid"]
    multi = args["multi"]
    n_layers = args["n_layers"]
    nogpu = args["nogpu"]
    precision = args["precision"]
    save_model = args["save_model"]
    seed = args["seed"]
    verbose = args["verbose"]
    problem_path = args["problem_path"]
    data_path = args["data_path"]

    # ------------------------------------------------------------------------

    # Configure TensorFlow.
    configure_tensorflow(args)

    # ------------------------------------------------------------------------

    # Import the problem to solve.
    if verbose:
        print(f"Importing python module for problem {problem_path}.")
    p = common.import_problem(problem_path)
    if debug:
        print(f"p = {p}")

    # Create the output directory under the current directory.
    output_dir = create_output_directory(p, args)
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
    XY_data = load_problem_data(args)
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
    models = load_or_create_models(p, args)
    if debug:
        print(f"models = {models}")

    # ------------------------------------------------------------------------

    # Create the optimizer to use for training.
    optimizer = create_optimizer(args)
    if debug:
        print(f"optimizer = {optimizer}")

    # ------------------------------------------------------------------------

    # Prepare inputs for TensorFlow.

    # Batch the training points as tf.Variable.
    # training_batches = create_batches(XY_data, args)
    # n_batches = len(training_batches)
    # if debug:
    #     print(f"training_batches = {training_batches}")
    #     print(f"n_batches = {n_batches}")

    # Convert data locations to tf.Variable.
    if verbose:
        print("Converting data locations to TensorFlow Variable.")
    X_data_tf = tf.Variable(X_data)
    if debug:
        print(f"X_data_tf = {X_data_tf}")

    # Convert data values to tf.Variable.
    if verbose:
        print("Converting data values to TensorFlow Variable.")
    Y_data_tf = tf.Variable(Y_data)
    if debug:
        print(f"Y_data_tf = {Y_data_tf}")

    # Count the data points.
    n_data = X_data_tf.shape[0]

    # -------------------------------------------------------------------------

    # Create loss histories by epoch and model as Python lists, so they can be
    # easily updated.
    # NOTE: CHANGE THIS TO A SINGLE NUMPY ARRAY.
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

            # Compute the model outputs at the training points.
            # Y_train_batch_per_model is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape (n_train, 1).
            Y_model = [model(X_data_tf) for model in models]
            if debug:
                print(f"Y_model = {Y_model}")

            # Compute the errors in the models at each training point.
            # E_model is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_train, 1).
            E_model = [
                Y_model[i] - tf.reshape(Y_data_tf[:, i], (n_data, 1))
                for i in range(p.n_var)
            ]
            if debug:
                print(f"E_model = {E_model}")

#             # Compute and save the individual loss functions for each model.
#             # The loss function is the RMS error.
#             # L_model is a list of Tensor objects.
#             # There are p.n_var Tensors in the list (one per model).
#             # Each Tensor has shape () (scalar).
#             L_model = [
#                 tf.math.sqrt(tf.reduce_sum(E**2)/n_train) for E in E_model
#             ]
#             if debug:
#                 print(f"L_model = {L_model}")
#             for i in range(p.n_var):
#                 varname = p.dependent_variable_names[i]
#                 loss[varname]['total'][epoch] = L_model[i].numpy()

#             # Compute and save the aggregate loss function.
#             # Tensor has shape () (scalar).
#             L = tf.reduce_sum(L_model)
#             if verbose:
#                 print(f"epoch = {epoch}, L = {L:.6E}")
#             loss['aggregate']['total'][epoch] = L.numpy()

#         # Compute the gradient of the loss wrt the network parameters.
#         # pgrad is a list of lists of Tensor objects.
#         # There are p.n_var sub-lists in the top-level list (one per
#         # model).
#         # There are 3 Tensors in each sub-list, with shapes based on
#         # model.trainable_variables.
#         pgrad = [
#             tape0.gradient(L, model.trainable_variables) for model in models
#         ]
#         if debug:
#             print(f"pgrad = {pgrad}")

#         # Update the parameters for this epoch.
#         for (g, m) in zip(pgrad, models):
#             optimizer.apply_gradients(zip(g, m.trainable_variables))

#         # Save the trained models.
#         if save_model > 0 and epoch % save_model == 0:
#             for (i, model) in enumerate(models):
#                 path = os.path.join(
#                     output_dir, 'models', f"{epoch}",
#                     f"model_{p.dependent_variable_names[i]}"
#                 )
#                 model.save(path)

        if debug:
            print(f"Ending epoch {epoch}.")

    # End of training loop.

    # Count the last epoch.
    n_epochs = epoch + 1
    if debug:
        print(f"n_epochs = {n_epochs}")

    # Print short training summary.
    t_stop = datetime.datetime.now()
    t_elapsed = t_stop - t_start
    if verbose:
        print(f"Total training time: {t_elapsed.total_seconds()} seconds")
        print(f"Epochs: {n_epochs}")
        # print(f"Final value of loss function: {L}")

    # Save the final trained models and descriptions.
    if save_model != 0:
        if multi:
            path = os.path.join(
                output_dir, "models", f"{epoch:06d}", "model_multi"
            )
            model = models[0]
            model.save(path)
            path = os.path.join(output_dir, "models", "model_multi.txt")
            old_stdout = sys.stdout
            with open(path, "w", encoding="utf-8") as f:
                sys.stdout = f
                model.summary()
            sys.stdout = old_stdout
        else:
            for (i, model) in enumerate(models):
                path = os.path.join(
                    output_dir, "models", f"{epoch:06d}",
                    f"model_{p.dependent_variable_names[i]}"
                )
                model.save(path)
                variable_name = p.dependent_variable_names[i]
                path = os.path.join(output_dir, "models",
                                    f"model_{variable_name}.txt")
                old_stdout = sys.stdout
                with open(path, "w", encoding="utf-8") as f:
                    sys.stdout = f
                    model.summary()
                sys.stdout = old_stdout

#     # Save the loss histories.
#     for v in p.dependent_variable_names:
#         path = os.path.join(output_dir, f"L_{v}.dat")
#         np.savetxt(path, loss[v]['total'])
#     path = os.path.join(output_dir, 'L.dat')
#     np.savetxt(path, loss['aggregate']['total'])


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
