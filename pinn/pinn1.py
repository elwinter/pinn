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

# Program defaults

# Default activation function to use in hidden nodes
DEFAULT_ACTIVATION = "sigmoid"

# Default learning rate
DEFAULT_LEARNING_RATE = 0.01

# Default maximum number of training epochs
DEFAULT_MAX_EPOCHS = 100

# Default number of hidden nodes per layer
DEFAULT_N_HID = 10

# Default number of layers in the fully-connected network, each with n_hid
# nodes
DEFAULT_N_LAYERS = 1

# Default TensorFlow precision for computations
DEFAULT_PRECISION = "float32"

# Default interval (in epochs) for saving the model
# 0 = do not save model
# -1 = only save at end
# n > 0: Save after every n epochs
DEFAULT_SAVE_MODEL = -1

# Default random number generator seed
DEFAULT_SEED = 0

# Default normalized weight to apply to the data loss function
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
        help="Specify activation function (default: %(default)s)"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
        help="Initial learning rate for training (default: %(default)s)"
    )
    parser.add_argument(
        "--load_model", default=None,
        help="Path to directory containing models to load (default:"
             " %(default)s)"
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
        help="Disable TensorFlow use of GPU(s) (default: %(default)s)"
    )
    parser.add_argument(
        "--precision", type=str, default=DEFAULT_PRECISION,
        help="Precision to use in TensorFlow solution (default: %(default)s)"
    )
    parser.add_argument(
        "--save_model", type=int, default=DEFAULT_SAVE_MODEL,
        help="Save interval (epochs) for trained model (0 = do not save, "
        "-1 = save at end, n > 0 = save every n epochs) (default: %(default)s)"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Seed for random number generator (default: %(default)s)"
    )
    parser.add_argument(
        "--use_constraints", action="store_true",
        help="Use constraint equations (if any) (default: %(default)s)."
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
        help="Path to problem description file (in python)"
    )
    parser.add_argument(
        "data_path",
        help="Path to problem data (IC, BC, etc.) file (default: %(default)s)"
    )
    parser.add_argument(
        "training_path",
        help="Path to training points file"
    )
    return parser


def main():
    """Begin main program."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}", flush=True)
    activation = args.activation
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
    verbose = args.verbose
    use_constraints = args.use_constraints
    w_data = args.w_data
    problem_path = args.problem_path
    data_path = args.data_path
    training_path = args.training_path

    # -------------------------------------------------------------------------

    # Configure TensorFlow.

    # If requested, disable TensorFlow use of GPU.
    if nogpu:
        if verbose:
            print("Disabling TensorFlow use of GPU.", flush=True)
        common.disable_gpus()

    # Set the backend TensorFlow precision.
    if verbose:
        print(f"Setting TensorFlow precision to {precision}.", flush=True)
    tf.keras.backend.set_floatx(precision)

    # Set the random number seed for reproducibility.
    if verbose:
        print(f"Seeding random number generator with {seed}.", flush=True)
    tf.random.set_seed(seed)

    # -------------------------------------------------------------------------

    # Read the problem description.

    # Import the problem to solve.
    if verbose:
        print(f"Importing python module for problem {problem_path}.", flush=True)
    p = common.import_problem(problem_path)
    if debug:
        print(f"p = {p}", flush=True)

    # Set up the output directory under the current directory.
    # The name of the output directory is the name of the problem python
    # module, with "-pinn1" appended to the end of the name.
    output_dir = os.path.join(".", f"{p.__name__}-pinn1")
    if debug:
        print(f"output_dir = {output_dir}", flush=True)
    os.mkdir(output_dir)

    # Record system information, model parameters, and problem definition,
    # data, and training grid.
    if verbose:
        print("Recording system information, model hyperparameters, and "
              "problem definition and data.", flush=True)
    common.save_system_information(output_dir)
    common.save_hyperparameters(args, output_dir)
    shutil.copy(problem_path, output_dir)
    shutil.copy(data_path, output_dir)
    shutil.copy(training_path, output_dir)

    # -------------------------------------------------------------------------

    # Read the training points.

    # These are just coordinate tuples, one per line, space-delimited.
    if verbose:
        print(f"Reading training points from {training_path}.", flush=True)
    # X_train is np.ndarray of shape (n_train, p.n_dim) OR (n_train,) for 1D.
    X_train = np.loadtxt(training_path, dtype=precision)
    if debug:
        print(f"X_train = {X_train}", flush=True)

    # If the data shape is 1-D (only one dimension), reshape to 2-D,
    # (n_train, 1) to make compatible with later TensorFlow calls, which
    # expect a 2D Tensor.
    if len(X_train.shape) == 1:
        if verbose:
            print("Training points are 1-D, reshaping to 2-D.")
        X_train = X_train.reshape(X_train.shape[0], 1)
        if debug:
            print(f"Reshaped X_train = {X_train}", flush=True)

    # Save a copy of the training data in the output directory.
    shutil.copy(training_path, os.path.join(output_dir, "X_train.dat"))

    # Count the training points.
    n_train = X_train.shape[0]
    if debug:
        print(f"n_train = {n_train}", flush=True)

    # -------------------------------------------------------------------------

    # Read the data points, which includes initial conditions, boundary
    # conditions, and any other data to be assimilated.

    # Each line contains a coordinate tuple, followed by a variables tuple,
    # containing the value of each variable at that location.
    if verbose:
        print(f"Reading training data from {data_path}.", flush=True)
    # Shape is (n_data, p.n_dim + p.n_var)
    XY_data = np.loadtxt(data_path, dtype=precision)
    if debug:
        print(f"XY_data = {XY_data}", flush=True)

    # If the data shape is 1-D (only one dimension), reshape to 2-D,
    # (n_train, 1) to make compatible with later TensorFlow calls, which
    # expect a 2D Tensor.
    if len(XY_data.shape) == 1:
        if verbose:
            print("Additional data is 1-D, reshaping to 2-D.", flush=True)
        XY_data = XY_data.reshape(1, XY_data.shape[0])
        if debug:
            print(f"Reshaped XY_data = {XY_data}", flush=True)

    # Save a copy of the additional data in the output directory.
    shutil.copy(data_path, os.path.join(output_dir, "XY_data.dat"))

    # Get the count of training data points.
    n_data = XY_data.shape[0]
    if debug:
        print(f"n_data = {n_data}", flush=True)

    # Extract the *locations* of the supplied data points.
    # Shape is (n_data, p.n_dim)
    X_data = XY_data[:, :p.n_dim]
    if debug:
        print(f"X_data = {X_data}", flush=True)

    # Extract the *values* of the supplied data points.
    # Shape is (n_data, p.n_var)
    Y_data = XY_data[:, p.n_dim:]
    if debug:
        print(f"Y_data = {Y_data}", flush=True)

    # -------------------------------------------------------------------------

    # Compute weights for residual and data loss functions.
    w_res = 1.0 - w_data
    if debug:
        print(f"w_res = {w_res}", flush=True)
        print(f"w_data = {w_data}", flush=True)

    # -------------------------------------------------------------------------

    # Create a model for each differential equation.
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

    # -------------------------------------------------------------------------

    # Create the optimizer to use for training.
    if verbose:
        print("Creating Adam optimizer.", flush=True)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    if debug:
        print(f"optimizer = {optimizer}", flush=True)

    # -------------------------------------------------------------------------

    # Prepare inputs for TensorFlow.

    # Convert training points to tf.Variable.
    if verbose:
        print("Converting training points to TensorFlow Variable", flush=True)
    X_train_tf = tf.Variable(X_train)
    if debug:
        print(f"X_train_tf = {X_train_tf}", flush=True)

    # Convert data locations to tf.Variable.
    if verbose:
        print("Converting data locations to TensorFlow Variable", flush=True)
    X_data_tf = tf.Variable(X_data)
    if debug:
        print(f"X_data_tf = {X_data_tf}", flush=True)

    # Convert data values to tf.Variable.
    if verbose:
        print("Converting data values to TensorFlow Variable", flush=True)
    Y_data_tf = tf.Variable(Y_data)
    if debug:
        print(f"Y_data_tf = {Y_data_tf}", flush=True)

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
    loss["aggregate"]["constraint"] = []
    loss["aggregate"]["data"] = []
    loss["aggregate"]["total"] = []
    if debug:
        print(f"loss = {loss}", flush=True)

    # -------------------------------------------------------------------------

    # Train the models.

    # Training involves presenting the training points and data points
    # together, a total of max_epochs times.

    # Record the training start time.
    t_start = datetime.datetime.now()
    if verbose:
        print(f"Training started at {t_start}.", flush=True)

    # Main training loop
    for epoch in range(max_epochs):
        if debug:
            print(f"Starting epoch {epoch}.", flush=True)

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
                Y_train_model = [model(X_train_tf) for model in models]
                if debug:
                    print(f"Y_train_model = {Y_train_model}", flush=True)

                # Compute the network outputs at all data points. These
                # are the values of the dependent variables Y to use when
                # comparing to the supplied data.
                # Y_data_model is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (n_data, 1).
                Y_data_model = [model(X_data_tf) for model in models]
                if debug:
                    print(f"Y_data_model = {Y_data_model}", flush=True)

            # Compute the gradients of the network outputs wrt inputs for
            # the training points. These are the values of the partial
            # derivatives dY/dX to use in the differential equations G.
            # dY_dX_train_model is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape (n_train, p.n_dim).
            dY_dX_train_model = [tape1.gradient(Y, X_train_tf) for Y in Y_train_model]
            if debug:
                print(f"dY_dX_train_model = {dY_dX_train_model}", flush=True)

            # Compute the values of the differential equations at all
            # training points.
            # G_train_model is a list of Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape (n_train, 1).
            G_train_model = [f(X_train_tf, Y_train_model, dY_dX_train_model) for f in p.de]
            if debug:
                print(f"G_train_model = {G_train_model}", flush=True)

            # Compute the values of the constraint equations (if any) at all
            # training points.
            # NOTE: Constraints are not associated with models.
            # C_train is a list of Tensor objects.
            # There are p.n_constraint Tensors in the list (one per
            # constraint).
            # Each Tensor has shape (n_train, 1).
            if use_constraints:
                C_train = [f(X_train_tf, Y_train_model, dY_dX_train_model)
                                for f in p.constraints]
                if debug:
                    print(f"C_train = {C_train}", flush=True)

            # -----------------------------------------------------------------

            # Compute the loss function for the equation residuals at the
            # training points for each model.
            # L_res_per_model is a list of Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape () (scalar).
            L_res_per_model = [
                tf.math.sqrt(tf.reduce_sum(G**2)/n_train)
                for G in G_train_model
            ]
            if debug:
                print(f"L_res_per_model = {L_res_per_model}", flush=True)

            # Compute the loss function for the constraints (if any) at the
            # training points.
            # L_constraint_per_constraint is a list of Tensor objects.
            # There are p.n_nconstraint Tensors in the list (one per
            # constraint).
            # Each Tensor has shape () (scalar).
            if use_constraints:
                L_constraint_per_constraint = [
                    tf.math.sqrt(tf.reduce_sum(C**2)/n_train)
                    for C in C_train
                ]
                if debug:
                    print(f"L_constraint_per_constraint = {L_constraint_per_constraint}",
                          flush=True)

            # Compute the errors in the predicted values at the data points.
            # E_data_per_model is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_data, 1).
            E_data_per_model = [
                Y_data_model[i] - tf.reshape(Y_data_tf[:, i], (n_data, 1))
                for i in range(p.n_var)
            ]
            if debug:
                print(f"E_data_per_model = {E_data_per_model}", flush=True)

            # Compute the loss functions for the data points for each
            # model.
            # L_data_per_model is a list of Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape () (scalar).
            L_data_per_model = [
                tf.math.sqrt(tf.reduce_sum(E**2)/n_data) for E in E_data_per_model
            ]
            if debug:
                print(f"L_data_per_model = {L_data_per_model}", flush=True)

            # Compute the weighted aggregate loss function per model.
            L_per_model = [
                w_res*L1 + w_data*L2 for (L1, L2)
                in zip(L_res_per_model, L_data_per_model)
            ]
            if debug:
                print(f"L_per_model = {L_per_model}", flush=True)

            # Compute the aggregated residual loss function.
            L_res = tf.math.reduce_sum(L_res_per_model)
            if debug:
                print(f"L_res = {L_res}", flush=True)

            # Compute the aggregated constraint loss function.
            if use_constraints:
                L_constraint = tf.math.reduce_sum(L_constraint_per_constraint)
                if debug:
                    print(f"L_constraint = {L_constraint}", flush=True)

            # Compute the aggregated data loss function.
            L_data = tf.math.reduce_sum(L_data_per_model)
            if debug:
                print(f"L_data = {L_data}", flush=True)

            # Compute the weighted aggregate loss function.
            L = w_res*(L_res + L_constraint) + w_data*L_data
            if debug:
                print(f"L = {L}", flush=True)

            # Save the losses for this epoch.
            for (i, v) in enumerate(p.dependent_variable_names):
                loss[v]["residual"].append(L_res_per_model[i].numpy())
                loss[v]["data"].append(L_data_per_model[i].numpy())
                loss[v]["total"].append(L_per_model[i].numpy())
            loss["aggregate"]["residual"].append(L_res.numpy())
            loss["aggregate"]["constraint"].append(L_constraint.numpy())
            loss["aggregate"]["data"].append(L_data.numpy())
            loss["aggregate"]["total"].append(L.numpy())
            if debug:
                print(f"loss = {loss}", flush=True)

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
            print(f"pgrad = {pgrad}", flush=True)

        # Update the parameters for this epoch.
        for (g, m) in zip(pgrad, models):
            optimizer.apply_gradients(zip(g, m.trainable_variables))

        # --------------------------------------------------------------------

        if verbose:
            print(f"epoch = {epoch}, (L_res, L_constraint, L_data, L) = "
                  f"({L_res:6e}, {L_constraint:6e}, {L_data:6e}, {L:6e})", flush=True)

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

    # Save the final trained models and descriptions.
    if save_model != 0:
        for (i, model) in enumerate(models):
            path = os.path.join(
                output_dir, "models", f"{epoch:06d}",
                f"model_{p.dependent_variable_names[i]}"
            )
            model.save(path)
            variable_name = p.dependent_variable_names[i]
            path = os.path.join(output_dir, f"model_{variable_name}.txt")
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
        os.path.join(output_dir, "L_constraint.dat"), loss["aggregate"]["constraint"]
    )
    np.savetxt(
        os.path.join(output_dir, "L_data.dat"), loss["aggregate"]["data"]
    )
    np.savetxt(
        os.path.join(output_dir, "L.dat"), loss["aggregate"]["total"]
    )


if __name__ == "__main__":
    """Begin main program."""
    main()
