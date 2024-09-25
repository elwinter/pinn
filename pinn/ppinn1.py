#!/usr/bin/env python

"""Use PINNs to solve a set of coupled 1st-order PDE and parameters.

This program will use a set of Physics-Informed Neural Networks (PINNs) to
solve a set of coupled 1st-order PDEs and estimate associated parameters.

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
DESCRIPTION = "Solve a set of 1st-order PDE and parameters using PINNs."


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
    batch_size = args.get("batch_size", -1)
    clobber = args.get("clobber", False)
    debug = args.get("debug", False)
    learning_rate = args.get("learning_rate", 0.01)
    load_model = args.get("load_model", None)
    max_epochs = args.get("max_epochs", 0)
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

    # Create the output directory under the current directory.
    if verbose:
        print("Creating output directory.")
    output_dir = common.create_output_directory(p, "-ppinn1", clobber=clobber)

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

    # Count the training points.
    n_train = X_train.shape[0]

    # ------------------------------------------------------------------------

    # Load the data points, which specify the function values at each point.
    # The data is always returned as a 2-D numpy array of shape
    # (n_data, p.n_dim + p.n_var).
    if verbose:
        print(f"Reading problem data from {data_path}.")
    XY_data = common.load_problem_data(data_path, precision=precision)

    # Get the count of data points.
    n_data = XY_data.shape[0]

    # Extract the *locations* of the supplied data points.
    # Shape is (n_data, p.n_dim)
    X_data = XY_data[:, :p.n_dim]

    # Extract the *values* of the supplied data points.
    # Shape is (n_data, p.n_var)
    Y_data = XY_data[:, p.n_dim:]

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

    # Create models for the parameters.
    if verbose:
        print("Creating untrained parameter models.")
    pmodels = []
    for ip in range(p.n_param):
        model = common.build_model(
            args["n_layers"], args["n_hid"], args["activation"])
        pmodels.append(model)

    # ------------------------------------------------------------------------

    # Create the optimizer to use for training.
    if verbose:
        print("Creating optimizer for training.")
    optimizer = common.create_optimizer(learning_rate)

    # ------------------------------------------------------------------------

    # Randomize the data if needed.
    if randomize:
        if verbose:
            print("Randomizing order of training data.")
        np.random.shuffle(X_train)

    # Split the training points into batches of TensorFlow Variables.
    if verbose:
        print("Batching training points.")
    batches = common.create_batches(X_train, batch_size)

    # ------------------------------------------------------------------------

    # Create residual loss history by epoch, batch, and model.
    # Shape is (max_epochs, n_batches, p.n_var), where
    # there is one plane per epoch, one row per batch, and one column per
    # dependent variable.
    Lb = np.zeros((max_epochs, len(batches), p.n_var))

    # Create residual, data, and aggregate loss history by epoch.
    # Shape is (max_epochs, 3), where
    # there is one row per epoch, and one column each for L_res, L_dat, L.
    Le = np.zeros((max_epochs, 3))

    # ------------------------------------------------------------------------

    # Compute weights for residual and data loss functions.
    w_res = 1.0 - w_data

    # ------------------------------------------------------------------------

    # Prepare inputs for TensorFlow.

    # Convert data locations to tf.Variable.
    if verbose:
        print("Converting data locations to TensorFlow Variable")
    Xd = tf.Variable(X_data)

    # Convert data values to tf.Variable.
    if verbose:
        print("Converting data values to TensorFlow Variable")
    Yd = tf.Variable(Y_data)

    # ------------------------------------------------------------------------

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

        # Phase 1: Train using the training points.

        # Train using each batch for this epoch.
        for (ib, Xb) in enumerate(batches):
            if debug:
                print(f"Starting batch {ib}.")

            # Run the forward pass for this batch.
            # tape0 is for computing gradients wrt network parameters.
            # tape1 is for computing 1st-order derivatives of outputs wrt
            # inputs.
            with tf.GradientTape(persistent=True) as tape0:
                with tf.GradientTape(persistent=True) as tape1:

                    # Compute the model outputs at the training points in this
                    # batch.
                    # Ybm contains the modeled values of the dependent
                    # variables Y to use in the differential equations G.
                    # Ybm is a list of tf.Tensor objects.
                    # There are p.n_var Tensors in the list (one per model).
                    # Each Tensor has shape (batch_size, 1).
                    Ybm = [model(Xb) for model in models]

                    # Compute the parameter estimates at the training points
                    # in this batch.
                    # Pbm contains the modeled values of the parameters P to
                    # use in the differential equations G.
                    # Pbm is a list of tf.Tensor objects.
                    # There are p.n_param Tensors in the list (one per
                    # parameter).
                    # Each Tensor has shape (batch_size, 1).
                    Pbm = [model(Xb) for model in pmodels]

                    # End of tape1 context

                # Compute the gradients of the network outputs wrt inputs for
                # the training points. These are the values of the partial
                # derivatives dY/dX to use in the differential equations G.
                # dYbm_dXb is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (n_train, p.n_dim).
                dYbm_dXb = [tape1.gradient(Y, Xb) for Y in Ybm]

                # Compute the values of the differential equations at all
                # training points.
                # Gbm is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (batch_size, 1).
                Gbm = [f(Xb, Ybm, dYbm_dXb, Pbm) for f in p.de]

                # Compute the loss function for the equation residuals at the
                # training points for each model.
                # Lbm_res is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per equation).
                # Each Tensor has shape () (scalar).
                Lbm_res = [tf.math.sqrt(tf.reduce_sum(G**2)/G.shape[0])
                           for G in Gbm]

                # Compute the unweighted and weighted residual loss function
                # for this batch.
                Lb_res = tf.math.reduce_sum(Lbm_res)
                wLb_res = w_res*Lb_res

                # End of tape0 context

            # Compute the gradient of the weighted residual loss function wrt
            # the network parameters for this batch.
            # pgrad is a list of lists of Tensor objects.
            # There are p.n_var sub-lists in the top-level list (one per
            # model).
            # There are 3 Tensors in each sub-list, with shapes:
            # Input weights: (p.n_dim, H)
            # Input biases: (H,)
            # Output weights: (H, 1)
            # Each Tensor is shaped based on model.trainable_variables.
            pgrad = [tape0.gradient(wLb_res, model.trainable_variables)
                     for model in models]
            ppgrad = [tape0.gradient(wLb_res, model.trainable_variables)
                      for model in pmodels]

            # Update the parameters for this batch.
            for (g, m) in zip(pgrad, models):
                optimizer.apply_gradients(zip(g, m.trainable_variables))
            for (g, m) in zip(ppgrad, pmodels):
                optimizer.apply_gradients(zip(g, m.trainable_variables))

            # Save the residual loss for this epoch, batch, and model.
            for iv in range(p.n_var):
                Lb[epoch][ib][iv] = Lbm_res[iv].numpy()

            if debug:
                print(f"Ending batch {ib}.")

        # --------------------------------------------------------------------

        # Phase 2: Train using the data points.

        with tf.GradientTape(persistent=True) as tape0:
            # Compute the model outputs at all data points. These
            # are the values of the dependent variables Y to use when
            # comparing to the supplied data.
            # Ydm is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape (n_data, 1).
            Ydm = [model(Xd) for model in models]

            # Compute the errors in the predicted values at the data points.
            # Edm is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_data, 1).
            Edm = [Ydm[i] - tf.reshape(Yd[:, i], (n_data, 1))
                   for i in range(p.n_var)]

            # Compute the loss functions for the data points for each
            # model.
            # Ldm is a list of Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape () (scalar).
            Ldm = [tf.math.sqrt(tf.reduce_sum(E**2)/n_data) for E in Edm]

            # Compute the unweighted and weighted data loss function for this
            # epoch.
            L_data = tf.math.reduce_sum(Ldm)
            wL_data = w_data*L_data

            # End of tape0 context

        # Compute the gradient of the weighted data loss function wrt
        # the network parameters.
        # pgrad is a list of lists of Tensor objects.
        # There are p.n_var sub-lists in the top-level list (one per
        # model).
        # There are 3 Tensors in each sub-list, with shapes:
        # Input weights: (p.n_dim, H)
        # Input biases: (H,)
        # Output weights: (H, 1)
        # Each Tensor is shaped based on model.trainable_variables.
        pgrad = [tape0.gradient(wL_data, model.trainable_variables)
                 for model in models]

        # Update the parameters for this epoch.
        for (g, m) in zip(pgrad, models):
            optimizer.apply_gradients(zip(g, m.trainable_variables))

        # --------------------------------------------------------------------

        # Phase 3: Compute the final loss for this epoch.

        # Compute the residual loss over all batches.
        E2m = [0.0]*p.n_var
        for (ib, Xb) in enumerate(batches):
            with tf.GradientTape(persistent=True) as tape1:
                Ybm = [model(Xb) for model in models]
            dYbm_dXb = [tape1.gradient(Y, Xb) for Y in Ybm]
            Pbm = [model(Xb) for model in pmodels]
            Gbm = [f(Xb, Ybm, dYbm_dXb, Pbm) for f in p.de]
            E2bm = [tf.reduce_sum(G**2) for G in Gbm]
            for iv in range(p.n_var):
                E2m[iv] += E2bm[iv]
        Lm_res = [tf.math.sqrt(E/n_train) for E in E2bm]
        L_res = tf.reduce_sum(Lm_res)

        # Compute the final data loss.
        Ydm = [model(Xd) for model in models]
        Edm = [Ydm[i] - tf.reshape(Yd[:, i], (n_data, 1))
               for i in range(p.n_var)]
        Ldm = [tf.math.sqrt(tf.reduce_sum(E**2)/n_data) for E in Edm]
        L_data = tf.math.reduce_sum(Ldm)

        # Compute the final overall loss.
        L = w_res*L_res + w_data*L_data
        if verbose:
            print(f"epoch = {epoch}, (L_res, L_data, L) = "
                  f"({L_res:6e}, {L_data:6e}, {L:6e})")
        Le[epoch, 0] = L_res
        Le[epoch, 1] = L_data
        Le[epoch, 2] = L

        # --------------------------------------------------------------------

        # Phase 4: Save the trained models.
        if save_model > 0 and epoch % save_model == 0:
            for (i, model) in enumerate(models):
                path = os.path.join(
                    output_dir, "models", f"{epoch:06d}",
                    f"model_{p.dependent_variable_names[i]}"
                )
                model.save(path)
            for (i, model) in enumerate(pmodels):
                path = os.path.join(
                    output_dir, "models", f"{epoch:06d}",
                    f"parameter_{p.parameter_names[i]}"
                )
                model.save(path)

        if debug:
            print(f"Ending epoch {epoch}.")

        # End of training loop

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

        # Dependent variable models
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

        # Parameter models
        for (i, model) in enumerate(pmodels):
            path = os.path.join(
                output_dir, "models", f"{epoch:06d}",
                f"parameter_{p.parameter_names[i]}"
            )
            model.save(path)
            parameter_name = p.parameter_names[i]
            path = os.path.join(output_dir, "models",
                                f"parameter_{parameter_name}.txt")
            old_stdout = sys.stdout
            with open(path, "w", encoding="utf-8") as f:
                sys.stdout = f
                model.summary()
            sys.stdout = old_stdout

    # Save the loss histories.
    path = os.path.join(output_dir, "Lb")
    np.save(path, Lb)
    path = os.path.join(output_dir, "Le.dat")
    np.savetxt(path, Le)

    # <HACK>
    # Print final parameter estimates.
    print(f"c1 = {Pbm[0]}")


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
