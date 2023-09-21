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

# Default number of training samples in batch (-1 - use single batch).
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

# Default normalized weight to apply to the data loss function.
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
        "--activation", "-a", default=DEFAULT_ACTIVATION,
        help="Specify activation function (default: %(default)s)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Size of training batches ({DEFAULT_BATCH_SIZE} "
        "for single batch)  (default: %(default)s)"
    )
    parser.add_argument(
        "--clobber", action="store_true",
        help="Clobber existing output directory (default: %(default)s)."
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to optional input data file (default: %(default)s)."
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
        help="Learning rate for training (default: %(default)s)"
    )
    parser.add_argument(
        "--load_model", default=None,
        help="Path to directory containing models to load (default:"
             " %(default)s)."
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
        "--nogpu", action="store_true",
        help="Disable TensorFlow use of GPU(s) (default: %(default)s)."
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
        "--seed", type=int, default=DEFAULT_SEED,
        help="Seed for random number generator (default: %(default)s)"
    )
    parser.add_argument(
        "--validation", default=None,
        help="Path to optional validation point file (default: %(default)s)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "--w_data", "-w", type=float, default=DEFAULT_W_DATA,
        help="Normalized weight for data loss function "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "problem_path",
        help="Path to problem description file."
    )
    parser.add_argument(
        "training_points",
        help="Path to file containing training points."
    )
    return parser


def disable_gpus():
    """Tell TensorFlow not to use GPU.

    Tell TensorFlow not to use GPU.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError : If this code cannot disable a GPU.
    """
    # Disable all GPUS.
    tf.config.set_visible_devices([], "GPU")

    # Make sure the GPU were disabled.
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"


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
    if args.debug:
        print(f"args = {args}", flush=True)
    activation = args.activation
    batch_size = args.batch_size
    clobber = args.clobber
    data = args.data
    debug = args.debug
    learning_rate = args.learning_rate
    load_model = args.load_model
    max_epochs = args.max_epochs
    H = args.n_hid
    n_layers = args.n_layers
    nogpu = args.nogpu
    precision = args.precision
    save_model = args.save_model
    seed = args.seed
    validation = args.validation
    verbose = args.verbose
    w_data = args.w_data
    problem_path = args.problem_path
    training_points = args.training_points

    # -------------------------------------------------------------------------

    # Configure TensorFlow.

    # If requested, disable TensorFlow use of GPU.
    if nogpu:
        if verbose:
            print("Disabling TensorFlow use of GPU.", flush=True)
        disable_gpus()

    # Set the backend TensorFlow precision.
    if verbose:
        print(f"Setting TensorFlow precision to {precision}.", flush=True)
    tf.keras.backend.set_floatx(precision)

    # -------------------------------------------------------------------------

    # Set up the run.

    # Import the problem to solve.
    if verbose:
        print(f"Importing module for problem {problem_path}.", flush=True)
    p = import_problem(problem_path)
    if debug:
        print(f"p = {p}", flush=True)

    # Set up the output directory under the current directory.
    # An exception is raised if the directory already exists, unless the
    # clobber flag is set.
    output_dir = os.path.join(".", f"{p.__name__}-pinn1")
    if debug:
        print(f"output_dir = {output_dir}", flush=True)
    if os.path.exists(output_dir):
        # Output directory already exists. Overwrite if requested.
        if not clobber:
            raise FileExistsError(f"Output directory {output_dir} exists!")
    else:
        # Create the output directory.
        os.mkdir(output_dir)

    # Record system information, network parameters, and problem definition.
    if verbose:
        print("Recording system information, model hyperparameters, and "
              "problem definition.", flush=True)
    common.save_system_information(output_dir)
    common.save_hyperparameters(args, output_dir)
    common.save_problem_definition(p, output_dir)

    # -------------------------------------------------------------------------

    # Read inputs for the run.

    # Read the training points. These are just coordinates like (t, x, y, ...).
    # No values of the dependent variables are provided.
    if verbose:
        print(f"Reading training points from {training_points}.", flush=True)
    # X_train is np.ndarray of shape (n_train, p.n_dim) OR (n_train,) for 1D.
    X_train = np.loadtxt(training_points, dtype=precision)
    if debug:
        print(f"X_train = {X_train}", flush=True)
    # If the data shape is 1-D (only one dimension), reshape to 2-D,
    # (n_train, 1) to make compatible with later TensorFlow calls.
    if len(X_train.shape) == 1:
        X_train = X_train.reshape(X_train.shape[0], 1)
        if debug:
            print(f"Reshaped X_train = {X_train}", flush=True)
    np.savetxt(os.path.join(output_dir, "X_train.dat"), X_train)

    # Count the training points.
    n_train = X_train.shape[0]
    if debug:
        print(f"n_train = {n_train}", flush=True)

    # If provided, read additional training data, which includes coordinates
    # and values for dependent variables. This file is typically used to
    # provide initial and boundary conditions, and any other data to be used
    # in the solution.
    if data:
        if verbose:
            print(f"Reading additional training data from {data}.", flush=True)
        # Shape is (n_data, p.n_dim + p.n_var)
        XY_data = np.loadtxt(data, dtype=precision)
        if debug:
            print(f"XY_data = {XY_data}", flush=True)
        # If the data shape is 1-D (only one data point), reshape to 2-D,
        # to make compatible with later TensorFlow calls.
        if len(XY_data.shape) == 1:
            XY_data = XY_data.reshape(1, XY_data.shape[0])
            if debug:
                print(f"Reshaped XY_data = {XY_data}", flush=True)
        np.savetxt(os.path.join(output_dir, "XY_data.dat"), XY_data)

        # Extract the *locations* of the supplied data points.
        # Shape is (n_data, p.n_dim)
        X_data = XY_data[:, :p.n_dim]
        if debug:
            print(f"X_data = {X_data}", flush=True)

        # Get the count of training data points.
        n_data = XY_data.shape[0]
        if debug:
            print(f"n_data = {n_data}", flush=True)

    # If provided, read and count the validation points.
    if validation:
        if verbose:
            print(f"Reading validation points from {validation}.", flush=True)
        # Shape is (n_val, p.n_dim)
        X_val = np.loadtxt(validation, dtype=precision)
        if debug:
            print(f"X_val = {X_val}", flush=True)
        # If the data shape is 1-D (only one dimension), reshape to 2-D,
        # to make compatible with later TensorFlow calls.
        if len(X_val.shape) == 1:
            X_val = X_val.reshape(X_val.shape[0], 1)
            if debug:
                print(f"Reshaped X_val = {X_val}", flush=True)
        np.savetxt(os.path.join(output_dir, "X_val.dat"), X_val)

        # Get the count of validation points.
        n_val = X_val.shape[0]
        if debug:
            print(f"n_val = {n_val}", flush=True)

    # Compute the normalized weight for the equation residuals, based on the
    # value of the data weight.
    w_res = 1.0 - w_data
    if debug:
        print(f"w_res = {w_res}", flush=True)
        print(f"w_data = {w_data}", flush=True)

    # Load or build one model for each differential equation defined in the
    # problem.
    models = []
    if load_model:
        if verbose:
            print(f"Loading trained models from {load_model}.", flush=True)
        for (i, v) in enumerate(p.dependent_variable_names):
            if verbose:
                print(f"Loading model for {v}.", flush=True)
            path = os.path.join(load_model, f"model_{v}")
            if debug:
                print(f"path = {path}", flush=True)
            model = tf.keras.models.load_model(path)
            if debug:
                print(f"model = {model}", flush=True)
            models.append(model)
    else:
        if verbose:
            print("Creating untrained models.", flush=True)
        for (i, v) in enumerate(p.dependent_variable_names):
            if verbose:
                print(f"Creating model for {v}.", flush=True)
            model = common.build_model(n_layers, H, activation)
            if debug:
                print(f"model = {model}", flush=True)
            models.append(model)
    if debug:
        print(f"models = {models}", flush=True)

    # Create the optimizer.
    if verbose:
        print("Creating Adam optimizer.", flush=True)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    if debug:
        print(f"optimizer = {optimizer}", flush=True)

    # Set the random number seed for reproducibility.
    if verbose:
        print(f"Seeding random number generator with {seed}.", flush=True)
    tf.random.set_seed(seed)

    # -------------------------------------------------------------------------

    # Prepare inputs for TensorFlow.

    # If a single batch was requested, compute the batch size.
    if batch_size == -1:
        batch_size = n_train
        if debug:
            print(f"Computed batch_size = {batch_size}", flush=True)

    # Split the training data into randomized batches.
    # Convert each batch to a tf.Variable for use in gradients.
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
                  f"{i_start} to {i_stop}.", flush=True)
        batch_indices = training_point_indices[i_start:i_stop]
        batch_points = X_train[batch_indices, ...]
        # Save the batch points.
        path = os.path.join(output_dir, f"X_batch_{i_batch:04d}.dat")
        np.savetxt(path, batch_points)
        # Append the batch to the list of batches.
        batches.append(tf.Variable(batch_points))
        i_start = i_stop
        i_batch += 1
    if debug:
        print(f"batches = {batches}", flush=True)

    # Convert additional data locations to tf.Variable.
    if data:
        X_data = tf.Variable(X_data)
        if debug:
            print(f"TF Variable of X_data = {X_data}", flush=True)

    # Convert validation locations to tf.Variable.
    if validation:
        X_val = tf.Variable(X_val)
        if debug:
            print(f"TF Variable of X_val = {X_val}", flush=True)

    # Create loss histories.
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

    # Convert the complete set of training point to a TF Variable.
    X_train = tf.Variable(X_train)
    if debug:
        print(f"X_train = {X_train}")

    # -------------------------------------------------------------------------

    # Train the models.

    # Record the training start time.
    t_start = datetime.datetime.now()
    if verbose:
        print(f"Training started at {t_start}.", flush=True)

    # Run training for max_epochs.
    for epoch in range(max_epochs):
        if debug:
            print(f"Starting epoch {epoch}.", flush=True)

        # --------------------------------------------------------------------

        # Step 1: Train using the equation residuals for each batch.
        # Update model parameters after each batch. If there is more than one
        # batch, this is stochastic gradient descent training.
        if debug:
            print("Step 1: Training with equation residuals.", flush=True)
        for (i_batch, X_batch) in enumerate(batches):
            if debug:
                print(f"Starting batch {i_batch}.", flush=True)

            # Determine the length of this batch.
            n_batch = X_batch.shape[0]
            if debug:
                print(f"n_batch = {n_batch}", flush=True)

            # Run the forward pass for this batch.
            # tape0 is for computing gradients wrt network parameters.
            # tape1 is for computing 1st-order derivatives of outputs wrt
            # inputs.
            with tf.GradientTape(persistent=True) as tape0:
                with tf.GradientTape(persistent=True) as tape1:

                    # Compute the network outputs at all training points
                    # in this batch. These are the values Y of the dependent
                    # variables X to use in the differential equations G for
                    # this batch.
                    # Y_batch is a list of tf.Tensor objects.
                    # There are p.n_var Tensors in the list (one per model).
                    # Each Tensor has shape (n_batch, 1).
                    Y_batch = [model(X_batch) for model in models]
                    if debug:
                        print(f"Y_batch = {Y_batch}", flush=True)

                # Compute the gradients of the network outputs wrt inputs for
                # this batch. These are the values of the partial derivatives
                # dY/dX to use in the differential equations G for this batch.
                # dY_dX_batch is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (n_batch, p.n_dim).
                dY_dX_batch = [tape1.gradient(Y, X_batch) for Y in Y_batch]
                if debug:
                    print(f"dY_dX_batch = {dY_dX_batch}", flush=True)

                # Compute the values of the differential equations at all
                # points in this batch.
                # G_batch is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (n_batch, 1).
                G_batch = [f(X_batch, Y_batch, dY_dX_batch) for f in p.de]
                if debug:
                    print(f"G_batch = {G_batch}", flush=True)

                # Compute the loss function for the equation residuals at the
                # batch training points for each model.
                # Lm_res_batch is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape () (scalar).
                Lm_res_batch = [
                    tf.math.sqrt(tf.reduce_sum(G**2)/n_batch) for G in G_batch
                ]
                if debug:
                    print(f"Lm_res_batch = {Lm_res_batch}", flush=True)

                # Compute the unweighted and weighted residual loss over all
                # models for the batch.
                L_res_batch = tf.math.reduce_sum(Lm_res_batch)
                if debug:
                    print(f"L_res_batch = {L_res_batch}", flush=True)
                Lw_res_batch = w_res*L_res_batch
                if debug:
                    print(f"Lw_res_batch = {Lw_res_batch}", flush=True)

            # Compute the gradient of the *weighted* residual loss function
            # wrt the network parameters for this batch.
            # pgrad_batch is a list of lists of Tensor objects.
            # There are p.n_var sub-lists in the list (one per model).
            # There are n_layers + 3 Tensors in each sub-list, with shapes:
            # Input weights: (H, p.n_dim)
            # Input biases: (H,)
            # For each hidden layer:
            #   Hidden layer weights (H, H)
            #   Hidden layer biases (H,)
            # Output weights: (H, 1)
            # Each Tensor is shaped based on model.trainable_variables.
            pgrad_batch = [
                tape0.gradient(Lw_res_batch, model.trainable_variables)
                for model in models
            ]
            if debug:
                print(f"pgrad_batch = {pgrad_batch}", flush=True)

            # Update the parameters for this batch and epoch.
            for (g, m) in zip(pgrad_batch, models):
                optimizer.apply_gradients(zip(g, m.trainable_variables))

        # --------------------------------------------------------------------

        # Step 2: Train using any additional data points. This includes
        # initial and boundary conditions.
        if data:
            if debug:
                print("Step 2: Training with data.", flush=True)

            # Run the forward pass for the data points in a single batch.
            # tape0 is for computing gradients wrt network parameters.
            with tf.GradientTape(persistent=True) as tape0:

                # Compute the network outputs at the data points.
                # Y_data is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (n_data, 1).
                Y_data = [model(X_data) for model in models]
                if debug:
                    print(f"Y_data = {Y_data}", flush=True)

                # Compute the errors for the data points for each model.
                # Em_data is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list.
                # Each Tensor has shape (n_data, 1).
                Em_data = [
                    Y_data[i] - tf.reshape(XY_data[:, p.n_dim + i],
                                           (n_data, 1))
                    for i in range(p.n_var)
                ]
                if debug:
                    print(f"Em_data = {Em_data}", flush=True)

                # Compute the loss functions for the data points for each
                # model.
                # Lm_data is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape () (scalar).
                Lm_data = [
                    tf.math.sqrt(tf.reduce_sum(E**2)/n_data)
                    for E in Em_data
                ]
                if debug:
                    print(f"Lm_data = {Lm_data}", flush=True)

                # Compute the unweighted and weighted data loss.
                L_data = tf.reduce_sum(Lm_data)
                if debug:
                    print(f"L_data = {L_data}", flush=True)
                Lw_data = w_data*L_data
                if debug:
                    print(f"Lw_data = {Lw_data}", flush=True)

            # Compute the gradient of the weighted data loss wrt the network
            # parameters.
            # pgrad_data is a list of lists of Tensor objects.
            # There are p.n_var sub-lists in the top-level list (one per
            # model).
            # There are 3 Tensors in each sub-list, with shapes:
            # Input weights: (H, p.n_dim)
            # Input biases: (H,)
            # Output weights: (H, 1)
            # Each Tensor is shaped based on model.trainable_variables.
            pgrad_data = [
                tape0.gradient(Lw_data, model.trainable_variables)
                for model in models
            ]
            if debug:
                print(f"pgrad_data = {pgrad_data}", flush=True)

            # Update the parameters for this data.
            for (g, m) in zip(pgrad_data, models):
                optimizer.apply_gradients(zip(g, m.trainable_variables))

        # --------------------------------------------------------------------

        # Step 3: Compute the end-of-epoch loss functions.
        if debug:
            print("Step 3: Computing end-of-epoch loss.", flush=True)

        # Compute the end-of-epoch residual loss functions.
        # G2_sum contains the sum of G**2 for each model, summed over all
        # training points in all batches.
        with tf.GradientTape(persistent=True) as tape1:
            Y_train = [model(X_train) for model in models]
            if debug:
                print(f"Y_train = {Y_train}", flush=True)
        dY_dX_train = [tape1.gradient(Y, X_train) for Y in Y_train]
        if debug:
            print(f"dY_dX_train = {dY_dX_train}", flush=True)
        G_train = [f(X_train, Y_train, dY_dX_train) for f in p.de]
        if debug:
            print(f"G_train = {G_train}", flush=True)
        G2_train_sum = np.array([tf.math.reduce_sum(G**2) for G in G_train])
        if debug:
            print(f"G2_train_sum = {G2_train_sum}", flush=True)
        Lm_res = tf.math.sqrt(G2_train_sum/n_train)
        for (i, v) in enumerate(p.dependent_variable_names):
            loss[v]["residual"].append(Lm_res[i])
        L_res = tf.reduce_sum(Lm_res)
        if debug:
            print(f"L_res = {L_res}", flush=True)
        loss["aggregate"]["residual"].append(L_res)

        # Compute the end-of-epoch data loss function.
        Y_data = [model(X_data) for model in models]
        if debug:
            print(f"Y_data = {Y_data}", flush=True)
        Em_data = [
            Y_data[i] - tf.reshape(XY_data[:, p.n_dim + i], (n_data, 1))
            for i in range(p.n_var)
        ]
        if debug:
            print(f"Em_data = {Em_data}", flush=True)
        Lm_data = [
            tf.math.sqrt(tf.reduce_sum(E**2)/n_data)
            for E in Em_data
        ]
        if debug:
            print(f"Lm_data = {Lm_data}", flush=True)
        for (i, v) in enumerate(p.dependent_variable_names):
            loss[v]["data"].append(Lm_data[i])
        L_data = tf.reduce_sum(Lm_data)
        if debug:
            print(f"L_data = {L_data}", flush=True)
        loss["aggregate"]["data"].append(L_data)

        # Compute the weighted model and aggregate loss.
        Lm = np.zeros(p.n_var)
        for i in range(p.n_var):
            Lm[i] = w_res*Lm_res[i] + w_data*Lm_data[i]
        if debug:
            print(f"Lm = {Lm}", flush=True)
        for (i, v) in enumerate(p.dependent_variable_names):
            loss[v]["total"].append(Lm[i])
        L = w_res*L_res + w_data*L_data
        if debug:
            print(f"L = {L}", flush=True)
        loss["aggregate"]["total"].append(L)

        if verbose:
            print(f"epoch = {epoch}, (L_res, L_data, L) = "
                  f"({L_res:6e}, {L_data:6e}, {L:6e})", flush=True)

        # Save the trained models.
        if save_model > 0 and epoch % save_model == 0:
            for (i, model) in enumerate(models):
                path = os.path.join(
                    output_dir, "models", f"{epoch:06d}",
                    f"model_{p.dependent_variable_names[i]}"
                )
                model.save(path)

        if debug:
            print(f"Ending epoch {epoch}.", flush=True)

    # Count the last epoch.
    n_epochs = epoch + 1
    if debug:
        print(f"n_epochs = {n_epochs}", flush=True)

    # Record the training end time.
    t_stop = datetime.datetime.now()
    t_elapsed = t_stop - t_start
    if verbose:
        print(f"Training stopped at {t_stop}.", flush=True)
        print(f"Total training time: {t_elapsed.total_seconds()} seconds",
              flush=True)
        print(f"Epochs: {n_epochs}", flush=True)
        print(f"Final value of loss function: {L}", flush=True)

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

    # Save the final trained models.
    if save_model != 0:
        for (i, model) in enumerate(models):
            path = os.path.join(
                output_dir, "models", f"{epoch:06d}",
                f"model_{p.dependent_variable_names[i]}"
            )
            model.save(path)


if __name__ == "__main__":
    """Begin main program."""
    main()
