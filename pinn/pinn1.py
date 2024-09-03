#!/usr/bin/env python

"""Use PINNs to solve a set of coupled 1st-order PDE.

This program will use a set of Physics-Informed Neural Networks (PINNs) to
solve a set of coupled 1st-order PDEs.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard Python modules.
import argparse
import copy
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
        "--w_data", "-w", type=float, default=0.5,
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
    directory is the name of the problem python module, with "-pinn1"
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
    output_dir = os.path.join(".", f"{p.__name__}-pinn1")
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

    Load the problem data, which includes initial conditions, boundary
    conditions, and any other data to be used in the solution.

    Each line contains a tuple of coordinates (the independent variable values
    for a location), followed by a tuple of dependent variable values,
    containing the values of each dependent variable at that location.

    Note that this scheme currently only supports Dirichlet boundary
    conditions.

    Parameters
    ----------
    args : dict
        Dictionary of command-line options.

    Returns
    -------
    XY_data : np.ndarray, shape(n_data, p.n_dim + p.n_var)
        Array of training points.

    Raises
    ------
    None
    """
    if args["verbose"]:
        print(f"Reading problem data from {args['data_path']}.")
    XY_data = np.loadtxt(args['data_path'], dtype=args["precision"])
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


def load_training_points(args: dict):
    """Load the training points.

    Load the training points. The training points are just coordinate tuples,
    one tuple per line, space-delimited.

    Parameters
    ----------
    args : dict
        Dictionary of command-line options.

    Returns
    -------
    X_train : np.ndarray, shape(n_train, p.n_dim)
        Array of training points.

    Raises
    ------
    None
    """
    if args["verbose"]:
        print(f"Reading training points from {args['training_path']}.")
    X_train = np.loadtxt(args["training_path"], dtype=args["precision"])
    if args["debug"]:
        print(f"X_train = {X_train}")

    # If the training point shape is 1-D (only one dimension), reshape to
    # 2-D, (n_train, 1) to make compatible with later TensorFlow calls, which
    # expect a 2D Tensor.
    if len(X_train.shape) == 1:
        if args["verbose"]:
            print("Training points are 1-D, reshaping to 2-D.")
        X_train = X_train.reshape(X_train.shape[0], 1)
        if args["debug"]:
            print(f"Reshaped X_train = {X_train}")

    # Return the training points.
    return X_train


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
        If --multi is specified.
        If --load_model is specified.
    """
    models = []
    if args["multi"]:
        raise TypeError("No --multi yet!")
#         if load_model:
#             raise TypeError(
#                 "Loading trained multi-output model not implemented!"
#             )
#         else:
#             if verbose:
#                 print("Creating untrained multi-output model.")
#             model = common.build_multi_output_model(n_layers, H, activation,
#                                                     p.n_var)
#             if debug:
#                 print(f"model = {model}")
#             models.append(model)
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


def forward_pass_1(models: list, X_train_batch_tf: tf.Variable, args: dict):
    """Perform forward pass 1 using a batch of training points.

    Perform forward pass 1 using a batch of training points.

    Compute the model outputs for this batch. These are the values of the
    dependent variables Y to use in the differential equations G.

    Parameters
    ----------
    models : list of keras.src.engine.sequential.Sequential
        Models to evaluate.
    X_train_batch_tf : tf.Variable, shape (batch_size, p.n_dim)
        tf.Variable of training points for this batch.
    args : dict
        Dictionary of command-line arguments.

    Returns
    -------
    Y_train_batch_per_model : list of tf.Variable, shape (batch_size, 1)
        Value of each model at each point in batch.

    Raises
    ------
    None
    """
    Y_train_batch_per_model = []
    if args["multi"]:
        raise TypeError("No --multi yet!")
        # For a multi-output network, repackage the results
        # into a list of Tensor for the individual variables.
        # Y_multi_batch_tf = models[0](X_train_batch_tf)
        # Y_train_batch_per_model = [
        #     tf.reshape(Y_multi_batch_tf[:, i], (n_train, 1))
        #     for i in range(p.n_var)
        # ]
    else:
        Y_train_batch_per_model = [model(X_train_batch_tf)
                                   for model in models]
    if args["debug"]:
        print(f"Y_train_batch_per_model = {Y_train_batch_per_model}")

    # Return the model values at each batch point.
    return Y_train_batch_per_model


def forward_pass_2(models: list, X_data_tf: tf.Variable, args: dict):
    """Perform forward pass 2 using th training data.

    Perform forward pass 2 using a batch of training data.

    Compute the model outputs for the training data. These are the values of
    the dependent variables Y to compare to the training data.

    Parameters
    ----------
    models : list of keras.src.engine.sequential.Sequential
        Models to evaluate.
    X_data_tf : tf.Variable, shape (n_data, p.n_dim)
        tf.Variable of data points for this batch.
    args : dict
        Dictionary of command-line arguments.

    Returns
    -------
    Y_data_per_model : list of tf.Variable, shape (n_dat, 1)
        Value of each model at each point in data.

    Raises
    ------
    None
    """
    Y_data_per_model = []
    if args["multi"]:
        raise TypeError("No --multi yet!")
        # For a multi-output network, repackage the results
        # into a list of Tensor for the individual variables.
        # Y_multi_data = models[0](X_data_tf)
        # Y_data_per_model = [tf.reshape(Y_multi_data[:, i], (n_data, 1))
        #                 for i in range(p.n_var)]
    else:
        Y_data_per_model = [model(X_data_tf) for model in models]

    # Return the model values at the data locations.
    return Y_data_per_model


def pinn1(args: dict):
    """Primary entry point for 1st-order PINN code.

    Use a 1st-order PINN to solve a set of differential equations.

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
    TypeError
        If batch size is greater than training set size.
        If --multi is specified.
    """
    # Extract the command-line arguments.
    if args["debug"]:
        print(f"args = {args}")
    batch_size = args["batch_size"]
    debug = args["debug"]
    max_epochs = args["max_epochs"]
    multi = args["multi"]
    save_model = args["save_model"]
    verbose = args["verbose"]
    w_data = args["w_data"]
    problem_path = args["problem_path"]
    data_path = args["data_path"]
    training_path = args["training_path"]

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

    # Save the problem definition, data, and training points.
    if verbose:
        print("Saving problem definition, data, and training points.")
    shutil.copy(problem_path, output_dir)
    shutil.copy(data_path, output_dir)
    shutil.copy(training_path, output_dir)

    # Save copies of the data and training points under standard names.
    path = os.path.join(output_dir, "XY_data.dat")
    shutil.copy(data_path, path)
    path = os.path.join(output_dir, "X_train.dat")
    shutil.copy(training_path, path)

    # ------------------------------------------------------------------------

    # Load the data points, which includes initial conditions, boundary
    # conditions, and any other data to be used in the solution.
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

    # Read the training points.
    X_train = load_training_points(args)
    if debug:
        print(f"X_train = {X_train}")

    # Count the training points.
    n_train = X_train.shape[0]
    if debug:
        print(f"n_train = {n_train}")

    # Make sure the batch size is <= number of training points.
    if batch_size > n_train:
        raise TypeError(f"Batch size ({batch_size}) must be <= number of "
                        f"training points ({n_train})!")

    # ------------------------------------------------------------------------

    # Compute weights for residual and data loss functions.
    w_res = 1.0 - w_data
    if debug:
        print(f"w_res = {w_res}")
        print(f"w_data = {w_data}")

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
    training_batches = create_batches(X_train, args)
    n_batches = len(training_batches)
    if debug:
        print(f"training_batches = {training_batches}")
        print(f"n_batches = {n_batches}")

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

    # Training involves presenting the training points and data points
    # to each model, a total of max_epochs times. Each epoch is composed of
    # all of the batches of the training points, and the single batch of
    # additional data. The model parameters are adjusted after each batch is
    # processed.

    # Variables are labelled according to source:
    # _train : computed using training points
    # _data : computed using data points
    # _model : computed using model
    # _batch : computed for the current batch
    # _epoch : computed for the current batch
    # _tf : A TensorFlow Variable

    # NOTE: In the current version of the code, the number of equations being
    # solved must be equal to the number of dependent variables in the set of
    # equations being solved. Both of these values are the same as the number
    # of models being trained. In other words:
    # len(p.de) = p.n_var = len(models)

    # Record the training start time.
    t_start = datetime.datetime.now()
    if verbose:
        print(f"Training started at {t_start}.")

    # Main training loop
    for epoch in range(max_epochs):
        if debug:
            print(f"Starting epoch {epoch}.")

        # --------------------------------------------------------------------

        # Create the list to hold the lists of unweighted and weighted
        # residual losses for each batch of training points.
        # Each list will contain n_batches elements, each of which is a list of
        # p.n_var elements (one element per equation)
        # The elements of the lowest-level list contain TensorFlow Variables
        # of shape (1,).
        L_res_per_batch_per_eqn = [None]*n_batches
        wL_res_per_batch_per_eqn = [None]*n_batches

        # --------------------------------------------------------------------

        # Part 1: Process each batch of training points for this epoch.
        for i_batch in range(n_batches):
            if debug:
                print(f"i_batch = {i_batch}")

            # X_train_batch_tf is a tf.Variable containing the training points
            # for this batch. It has shape (batch_size, p.n_dim). The value of
            # batch_size may be less than the specified batch size for the last
            # batch, if n_train is not an integer multiple of batch_size.
            X_train_batch_tf = training_batches[i_batch]
            if debug:
                print(f"X_train_batch_tf = {X_train_batch_tf}")

            # Run the forward pass of each model for this training batch.
            # tape0 is for computing gradients wrt network parameters.
            # tape1 is for computing 1st-order derivatives of outputs wrt
            # inputs.
            with tf.GradientTape(persistent=True) as tape0:
                with tf.GradientTape(persistent=True) as tape1:

                    # Compute the model outputs for this batch. These are the
                    # values of the dependent variables Y to use in the
                    # differential equations G.
                    # Y_train_batch_per_model is a list of tf.Tensor objects
                    # containing the model outputs at each point in the batch.
                    # There are p.n_var Tensors in the list (one per model).
                    # Each Tensor has shape (n, 1).
                    Y_train_batch_per_model = forward_pass_1(
                        models, X_train_batch_tf, args
                    )
                    if debug:
                        print("Y_train_batch_per_model = "
                              f"{Y_train_batch_per_model}")

                    # End of tape1 context.

                # Compute the gradients of the model outputs wrt inputs for
                # the training points in this batch. These are the values of
                # the partial derivatives dY/dX to use in the differential
                # equations G.
                # dY_dX_train_batch_per_model is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (batch_size, p.n_dim).
                dY_dX_train_batch_per_model = [
                    tape1.gradient(Y, X_train_batch_tf)
                    for Y in Y_train_batch_per_model
                ]
                if debug:
                    print("dY_dX_train_batch_per_model = "
                          f"{dY_dX_train_batch_per_model}")

                # Compute the values of the differential equations at all
                # training points in the batch.
                # G_train_batch_per_eqn is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per equation).
                # Each Tensor has shape (batch_size, 1).
                G_train_batch_per_eqn = [
                    f(X_train_batch_tf, Y_train_batch_per_model,
                      dY_dX_train_batch_per_model) for f in p.de
                ]
                if debug:
                    print(f"G_train_batch_per_eqn = {G_train_batch_per_eqn}")

                # Compute the unweighted loss function for the equation
                # residuals at the training points in this batch for each
                # equation.
                # L_res_batch_per_eqn is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per equation).
                # Each Tensor has shape () (scalar).
                L_res_batch_per_eqn = [
                    tf.math.sqrt(tf.reduce_sum(G**2)/len(G))
                    for G in G_train_batch_per_eqn
                ]
                if debug:
                    print(f"L_res_batch_per_eqn = {L_res_batch_per_eqn}")

                # Save a copy of the unweighted residual losses for each
                # equation for this batch.
                L_res_per_batch_per_eqn[i_batch] = copy.deepcopy(
                    L_res_batch_per_eqn)
                if debug:
                    print("L_res_per_batch_per_eqn = "
                          f"{L_res_per_batch_per_eqn}")

                # Compute the weighted loss function for the equation residuals
                # at the training points in this batch for each model.
                # wL_res_batch_per_eqn is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per equation).
                # Each Tensor has shape () (scalar).
                wL_res_batch_per_eqn = [L*w_res for L in L_res_batch_per_eqn]
                if debug:
                    print(f"wL_res_batch_per_eqn = {wL_res_batch_per_eqn}")

                # Save a copy of the weighted residual losses for each model
                # for this batch.
                wL_res_per_batch_per_eqn[i_batch] = copy.deepcopy(
                    wL_res_batch_per_eqn)
                if debug:
                    print("wL_res_per_batch_per_eqn = "
                          f"{wL_res_per_batch_per_eqn}")

                # Compute the aggregated weighted residual loss function for
                # all equations for this batch.
                wL_res_batch = tf.math.reduce_sum(wL_res_batch_per_eqn)
                if debug:
                    print(f"wL_res_batch = {wL_res_batch}")

                # End of tape0 context.

            # Compute the gradient of the aggregated weighted residual loss
            # wrt the network parameters.
            # pgrad is a list of lists of Tensor objects.
            # There are p.n_var lists in the top-level list (one per model).
            # There are 3 Tensors in each sub-list.
            # Each Tensor is shaped based on model.trainable_variables.
            pgrad = [
                tape0.gradient(wL_res_batch, model.trainable_variables)
                for model in models
            ]
            if debug:
                print(f"pgrad = {pgrad}")

            # Update the model parameters for this epoch and batch.
            for (g, m) in zip(pgrad, models):
                optimizer.apply_gradients(zip(g, m.trainable_variables))

            if debug:
                print(f"epoch = {epoch}, batch {i_batch}: "
                      f"wL_res_batch = {wL_res_batch}")

            # End of all training point batches for this epoch.

        # --------------------------------------------------------------------

        # Part 2: Process the data points for this epoch.

        # Run the forward pass for the data points for this epoch.
        # tape0 is for computing gradients wrt network parameters.
        with tf.GradientTape(persistent=True) as tape0:
            Y_data_per_model = forward_pass_2(models, X_data_tf, args)
            if debug:
                print(f"Y_data_per_model = {Y_data_per_model}")

            # Compute the errors in the predicted values at the data points.
            # E_data_per_model is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list (one per dependent
            # variable).
            # Each Tensor has shape (n_data, 1).
            E_data_per_model = [
                Y_data_per_model[i] - tf.reshape(Y_data_tf[:, i], (n_data, 1))
                for i in range(p.n_var)
            ]
            if debug:
                print(f"E_data_per_model = {E_data_per_model}")

            # Compute the unweighted loss functions for the data points for
            # each model.
            # L_data_per_model is a list of Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape () (scalar).
            L_data_per_model = [
                tf.math.sqrt(tf.reduce_sum(E**2)/n_data)
                for E in E_data_per_model
            ]
            if debug:
                print(f"L_data_per_model = {L_data_per_model}")

            # Compute the weighted loss functions for the data points for each
            # model.
            # wL_data_per_model is a list of Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape () (scalar).
            wL_data_per_model = [w_data*L_data for L_data in L_data_per_model]
            if debug:
                print(f"wL_data_per_model = {wL_data_per_model}")

            # Compute the aggregated weighted data loss function.
            wL_data = tf.math.reduce_sum(wL_data_per_model)
            if debug:
                print(f"wL_data = {wL_data}")

            # End of tape0 context.

        # Compute the gradient of the aggregated weighted data loss
        # wrt the network parameters.
        # pgrad is a list of lists of Tensor objects.
        # There are p.n_var sub-lists in the top-level list (one per
        # model).
        # There are 3 Tensors in each sub-list, with shapes:
        # Input weights: (p.n_dim, H)
        # Input biases: (H,)
        # Output weights: (H, 1)
        # Each Tensor is shaped based on model.trainable_variables.
        pgrad = [
            tape0.gradient(wL_data, model.trainable_variables)
            for model in models
        ]
        if debug:
            print(f"pgrad = {pgrad}")

        # Update the network parameters for this epoch.
        for (g, m) in zip(pgrad, models):
            optimizer.apply_gradients(zip(g, m.trainable_variables))

        # --------------------------------------------------------------------

        # At this point, all of the training points, and all of the data
        # points, have been used to train the network for this epoch.

        # --------------------------------------------------------------------

        # Compute the overall loss function for the epoch.

        # Convert the individual per-batch per-model unweighted residual
        # losses back to sum of squared residuals. Then total them, and
        # compute the RMS residual over the entire training set for each
        # model.
        sum_G2_per_eq = [0.0]*len(models)
        for i_batch in range(n_batches):
            this_batch_size = training_batches[i_batch].shape[0]
            for i_model in range(len(models)):
                sum_G2_per_eq[i_model] += (
                    L_res_per_batch_per_eqn[i_batch][i_model]**2
                    * this_batch_size
                )
        if debug:
            print(f"sum_G2_per_eq = {sum_G2_per_eq}")
        L_res_per_model = [
            tf.math.sqrt(sum_G2/n_train) for sum_G2 in sum_G2_per_eq
        ]
        L_res = tf.reduce_sum(L_res_per_model)
        if debug:
            print(f"L_res = {L_res}")

        # Convert the weighted data loss to unweighted.
        if w_data > 0.0:
            L_data = wL_data/w_data
        else:
            L_data = 0.0
        if debug:
            print(f"L_data = {L_data}")

        # Compute the per-model weighted loss.
        L_per_model = [
            w_res*Lr + w_data*Ld
            for (Lr, Ld) in zip(L_res_per_model, L_data_per_model)
        ]
        if debug:
            print(f"L_per_model = {L_per_model}")

        # Compute the final weighted loss.
        L = w_res*L_res + w_data*L_data
        if debug:
            print(f"L = {L}")

        # Save the per-model losses.
        for (i, v) in enumerate(p.dependent_variable_names):
            loss[v]["residual"].append(L_res_per_model[i])
            loss[v]["data"].append(L_data_per_model[i])
            loss[v]["total"].append(L_per_model[i])

        # Save the aggregate losses.
        loss["aggregate"]["residual"].append(L_res)
        loss["aggregate"]["data"].append(L_data)
        loss["aggregate"]["total"].append(L)

        if verbose:
            print(f"Epoch = {epoch}: (L_res, L_data, L) = "
                  f"({L_res:.4e}, {L_data:.4e} {L:.4e})")

        # --------------------------------------------------------------------

        # Save the trained models.
        if save_model > 0 and epoch % save_model == 0:
            if multi:
                raise TypeError("No --multi yet!")
                # path = os.path.join(
                #     output_dir, "models", f"{epoch:06d}", "model_multi"
                # )
                # models[0].save(path)
            else:
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
        print(f"Total training time: {t_elapsed.total_seconds()} seconds")
        print(f"Epochs: {n_epochs}")
        print(f"Final value of loss function: {L}")

    # Save the final trained models and descriptions.
    if save_model != 0:
        if multi:
            raise TypeError("No --multi yet!")
            # path = os.path.join(
            #     output_dir, "models", f"{epoch:06d}", "model_multi"
            # )
            # model.save(path)
            # path = os.path.join(output_dir, "models", "model_multi.txt")
            # old_stdout = sys.stdout
            # with open(path, "w", encoding="utf-8") as f:
            #     sys.stdout = f
            #     model.summary()
            # sys.stdout = old_stdout
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
    pinn1(vars(args))


if __name__ == "__main__":
    main()
