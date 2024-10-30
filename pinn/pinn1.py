#!/usr/bin/env python

"""Use PINNs to solve a set of coupled 1st-order PDE.

This program will use a set of Physics-Informed Neural Networks (PINNs) to
solve a set of coupled 1st-order PDEs.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard Python modules.
import copy
import datetime
import os
import shutil

# Import 3rd-party modules.
import numpy as np
import tensorflow as tf

# Import project modules.
from pinn import common


# Program constants

# Program description
DESCRIPTION = "Solve a set of coupled 1st-order PDE using the PINN method."

# Default values for command-line arguments.
DEFAULT_ARGUMENTS = copy.deepcopy(common.DEFAULT_ARGUMENTS)
DEFAULT_ARGUMENTS["randomize"] = False
DEFAULT_ARGUMENTS["w_data"] = 0.5
DEFAULT_ARGUMENTS["problem_path"] = None
DEFAULT_ARGUMENTS["data_path"] = None
DEFAULT_ARGUMENTS["training_path"] = None


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
    parser = common.create_neural_network_command_line_argument_parser(
        DESCRIPTION)
    parser.add_argument(
        "--randomize", "-r", action="store_true",
        default=DEFAULT_ARGUMENTS["randomize"],
        help="Randomize the order of the training data (default: %(default)s)"
    )
    parser.add_argument(
        "--w_data", "-w", type=float,
        default=DEFAULT_ARGUMENTS["w_data"],
        help="Normalized weight for data loss function "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "problem_path",
        help="Path to problem description file (in python)"
    )
    parser.add_argument(
        "data_path",
        help="Path to problem data file"
    )
    parser.add_argument(
        "training_path",
        help="Path to training points file"
    )
    return parser


def create_output_directory(problem_name: str, clobber: bool):
    """Create the output directory for this problem.

    Create the output directory for this problem. The name of the output
    directory is the name of the problem python module, with "-pinn1"
    appended to the end of the name.

    Parameters
    ----------
    problem_name : str
        Problem name.
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
    output_dir = os.path.join(".", f"{problem_name}-pinn1")
    if os.path.isdir(output_dir) and clobber:
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    return output_dir


def pinn1(args: dict):
    """Primary entry point for 1st-order PINN code.

    Use a 1st-order PINN to solve a set of differential equations.

    Regarding variable names:

    X*: Contains values of independent variables.
    Y*: Contains values of dependent variables.
    XY*: Contains values of independent and dependent variables.

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
    # Use defaults for unspecified arguments. Merge in additional settings
    # which are passed in by the caller.
    local_args = copy.deepcopy(DEFAULT_ARGUMENTS)
    if args is not None:
        local_args.update(args)
    args = local_args

    # Local convenience variables
    activation = args["activation"]
    batch_size = args["batch_size"]
    clobber = args["clobber"]
    debug = args["debug"]
    learning_rate = args["learning_rate"]
    load_model = args["load_model"]
    max_epochs = args["max_epochs"]
    multi = args["multi"]
    n_hid = args["n_hid"]
    n_layers = args["n_layers"]
    nogpu = args["nogpu"]
    precision = args["precision"]
    randomize = args["randomize"]
    save_model = args["save_model"]
    seed = args["seed"]
    verbose = args["verbose"]
    w_data = args["w_data"]
    problem_path = args["problem_path"]
    data_path = args["data_path"]
    training_path = args["training_path"]

    # ------------------------------------------------------------------------

    # <HACK>
    if multi:
        raise TypeError("--multi not supported yet!")
    # </HACK>

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
    output_dir = create_output_directory(p.__name__, clobber)
    if debug:
        print(f"output_dir = {output_dir}")

    # Record system information and program arguments.
    if verbose:
        print("Saving system information and program arguments.")
    common.save_system_information(output_dir)
    common.save_arguments(args, output_dir)

    # Save the problem definition, data points, and training points.
    if verbose:
        print("Saving problem definition, data points, and training "
              "points.")
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

    # Load the training points.
    if verbose:
        print(f"Loading training points from {training_path}.")
    X_train = common.load_problem_data(training_path, precision)
    if debug:
        print(f"X_train = {X_train}")

    # Count the training points.
    n_train = X_train.shape[0]
    if debug:
        print(f"n_train = {n_train}")

    # Optionally randomize the order of the training points in-place.
    if randomize:
        if verbose:
            print("Shuffling training points.")
        np.random.shuffle(X_train)

    # ------------------------------------------------------------------------

    # Load or create PINN models for the variables.
    if load_model is not None:
        # <TODO> TEST THIS.
        models = common.load_models(load_model, p.dependent_variable_names,
                                    multi)
        # </TODO>
    else:
        models = common.create_models(p.dependent_variable_names, n_layers,
                                      n_hid, activation, multi)
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

    # Prepare training data for TensorFlow.

    # Convert independent and dependent variables to tf.Variable.
    Xd = tf.Variable(X_data)
    Yd = tf.Variable(Y_data)
    if debug:
        print(f"Xd = {Xd}")
        print(f"Yd = {Yd}")

    # Batch the training data as tf.Variable.
    # Xdbs is a list of tf.Variable.
    # Each tf.Variable has shape (<= batch_size, p.n_dim)
    # Ydbs is a list of tf.Variable.
    # Each tf.Variable has shape (<= batch_size, p.n_var)
    if verbose:
        print("Batching training data.")
    Xdbs = common.create_batches(X_data, batch_size)
    Ydbs = common.create_batches(Y_data, batch_size)
    nd_batches = len(Xdbs)
    if debug:
        print(f"Xdbs = {Xdbs}")
        print(f"Ydbs = {Ydbs}")
        print(f"nd_batches = {nd_batches}")

    # ------------------------------------------------------------------------

    # Prepare training points for TensorFlow.

    # Convert training points to tf.Variable.
    Xt = tf.Variable(X_train)
    if debug:
        print(f"Xt = {Xt}")

    # Batch the training points as tf.Variable.
    # Xtbs is a list of tf.Variable.
    # Each tf.Variable has shape (<= batch_size, p.n_dim)
    if verbose:
        print("Batching training points.")
    Xtbs = common.create_batches(X_train, batch_size)
    nt_batches = len(Xtbs)
    if debug:
        print(f"Xtbs = {Xtbs}")
        print(f"nt_batches = {nt_batches}")

    # ------------------------------------------------------------------------

    # Create loss histories by epoch, model, and type (residual, data, total).
    losses_res = np.zeros((max_epochs, p.n_var))
    losses_dat = np.zeros((max_epochs, p.n_var))
    losses = np.zeros((max_epochs, p.n_var))

    # ------------------------------------------------------------------------

    # Compute weights for residual and data loss functions.
    w_res = 1.0 - w_data
    if debug:
        print(f"w_res = {w_res}")
        print(f"w_data = {w_data}")

    # ------------------------------------------------------------------------

    # Train the models.

    # Training involves presenting the training points and data points
    # to each model, a total of max_epochs times. Each epoch is composed of
    # all of the batches of the training points, and all of the batches of
    # additional data. The model parameters are adjusted after each batch is
    # processed. The overall loss function for the epoch is computed after
    # all batches are processed.

    # Record the training start time.
    t_start = datetime.datetime.now()
    if verbose:
        print(f"Training started at {t_start}.")

    # Train for the maximum number of epochs.
    for epoch in range(max_epochs):
        if debug:
            print(f"Starting epoch {epoch}.")

        # --------------------------------------------------------------------

        # Part 1: Process each batch of training points for this epoch.
        for it_batch in range(nt_batches):
            if debug:
                print(f"Starting epoch {epoch}, training batch {it_batch}.")

            # Xtb is a tf.Variable containing the independent variable values
            # for this batch. It has shape (<= batch_size, p.n_dim).
            Xtb = Xtbs[it_batch]
            if debug:
                print(f"Xtb = {Xtb}")

            # Run the forward pass of each model for this batch.
            # tape0 is for computing gradients wrt network parameters.
            # tape1 is for computing 1st-order derivatives of outputs wrt
            # inputs.
            with tf.GradientTape(persistent=True) as tape0:
                with tf.GradientTape(persistent=True) as tape1:

                    # Compute the model outputs at the training points.
                    # Ymb is a list of tf.Tensor objects.
                    # There are p.n_var Tensors in the list (one per model).
                    # Each Tensor has shape (<= batch_size, 1).
                    Ymbs = [model(Xtb) for model in models]
                    if debug:
                        print(f"Ymbs = {Ymbs}")

                    # End of tape1 context.

                # Compute the gradients of the model outputs wrt inputs for
                # the training points in this batch. These are the values of
                # the partial derivatives dY/dX to use in the differential
                # equations G.
                # dYmb_dXtb is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (<= batch_size, p.n_dim).
                dYmb_dXtbs = [tape1.gradient(Y, Xtb) for Y in Ymbs]
                if debug:
                    print(f"dYmb_dXtbs = {dYmb_dXtbs}")

                # Compute the values of the differential equations at all
                # training points in the batch.
                # Gtbs is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per equation).
                # Each Tensor has shape (<= batch_size, 1).
                Gtbs = [f(Xtb, Ymbs, dYmb_dXtbs) for f in p.de]
                if debug:
                    print(f"Gtbs = {Gtbs}")

                # Compute the residual loss functions at the training points
                # in this batch for each equation.
                # Lresbs is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per equation).
                # Each Tensor has shape () (scalar).
                Lresbs = [tf.math.sqrt(tf.reduce_sum(G**2)/G.shape[0])
                          for G in Gtbs]
                if debug:
                    print(f"Lresbs = {Lresbs}")

                # Compute the weighted loss function for the equation residuals
                # at the training points in this batch for each model.
                # wLresbs is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per equation).
                # Each Tensor has shape () (scalar).
                wLresbs = [L*w_res for L in Lresbs]
                if debug:
                    print(f"wLresbs = {wLresbs}")

                # Compute the aggregated weighted residual loss function for
                # all equations for this batch.
                wLresb = tf.math.reduce_sum(wLresbs)
                if debug:
                    print(f"epoch = {epoch}, training batch = {it_batch}, "
                          f"wLresb = {wLresb}")

                # End of tape0 context.

            # Compute the gradient of the aggregated weighted residual loss
            # wrt the network parameters.
            # pgrad is a list of lists of Tensor objects.
            # There are p.n_var lists in the top-level list (one per model).
            # There are 3 Tensors in each sub-list.
            # Each Tensor is shaped based on model.trainable_variables.
            pgrad = [
                tape0.gradient(wLresb, model.trainable_variables)
                for model in models
            ]
            if debug:
                print(f"pgrad = {pgrad}")

            # Update the model parameters for this epoch and batch.
            for (g, m) in zip(pgrad, models):
                optimizer.apply_gradients(zip(g, m.trainable_variables))

            if debug:
                print(f"Ending epoch {epoch}, training batch {it_batch}.")

            # End of loop over traing point batches.

        # --------------------------------------------------------------------

        # Part 2: Train using the data points for this epoch.
        for id_batch in range(nd_batches):
            if debug:
                print(f"Starting epoch {epoch}, data batch {id_batch}.")

            # Xdb is a tf.Variable containing the independent variable values
            # for this batch. It has shape (<= batch_size, p.n_dim).
            Xdb = Xdbs[id_batch]
            Ydb = Ydbs[id_batch]
            if debug:
                print(f"Xdb = {Xdb}")
                print(f"Ydb = {Ydb}")

            # Run the forward pass for the data points for this batch.
            # tape0 is for computing gradients wrt network parameters.
            with tf.GradientTape(persistent=True) as tape0:

                # Compute the model outputs at the data points.
                # Ymbs is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (<= batch_size, 1).
                Ymbs = [model(Xdb) for model in models]
                if debug:
                    print(f"Ymbs = {Ymbs}")

                # Compute the errors in the predicted values at the data points.
                # Embs is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per dependent
                # variable).
                # Each Tensor has shape (n_data, 1).
                Embs = [
                    Ymbs[i] - tf.reshape(Ydb[:, i], (Ydb.shape[0], 1))
                    for i in range(p.n_var)
                ]
                if debug:
                    print(f"Embs = {Embs}")

                # Compute the data loss functions for the data points in this
                # batch each model.
                # Ldatbs is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape () (scalar).
                Ldatbs = [
                    tf.math.sqrt(tf.reduce_sum(E**2)/E.shape[0])
                    for E in Embs
                ]
                if debug:
                    print(f"Ldatbs = {Ldatbs}")

                # Compute the weighted loss functions for the data points for each
                # model.
                # wLdatbs is a list of Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape () (scalar).
                wLdatbs = [w_data*L_data for L_data in Ldatbs]
                if debug:
                    print(f"wLdatbs = {wLdatbs}")

                # Compute the aggregated weighted data loss function.
                wLdatb = tf.math.reduce_sum(wLdatbs)
                if debug:
                    print(f"epoch = {epoch}, data batch = {id_batch}, "
                          f"wLdatb = {wLdatb}")

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
                tape0.gradient(wLdatb, model.trainable_variables)
                for model in models
            ]
            if debug:
                print(f"pgrad = {pgrad}")

            # Update the model parameters for this epoch and batch.
            for (g, m) in zip(pgrad, models):
                optimizer.apply_gradients(zip(g, m.trainable_variables))

            if debug:
                print(f"Ending epoch {epoch}, data batch {id_batch}.")

            # End of loop over training data batches.

        # --------------------------------------------------------------------

        # At this point, all of the training points, and all of the data
        # points, have been used to train the network for this epoch.

        # --------------------------------------------------------------------

        # Part 3: Compute the end-of-epoch loss.

        # Part 3a: Compute residual loss.
        sum_G2 = np.zeros(p.n_var)
        for it_batch in range(nt_batches):
            Xtb = Xtbs[it_batch]
            with tf.GradientTape(persistent=True) as tape1:
                Ymbs = [model(Xtb) for model in models]
                # End of tape1 context.
            dYmb_dXtbs = [tape1.gradient(Y, Xtb) for Y in Ymbs]
            Gtbs = [f(Xtb, Ymbs, dYmb_dXtbs) for f in p.de]
            sum_G2s = np.array([tf.reduce_sum(G**2).numpy() for G in Gtbs])
            sum_G2 += sum_G2s
            # End of training point batches.
        Lress = np.array([np.sqrt(G2/n_train) for G2 in sum_G2])
        Lres = np.sum(Lress)
        losses_res[epoch] = Lress
        losses_res[epoch][-1] = Lres

        # Part 3b: Compute data loss.
        sum_E2 = np.zeros(p.n_var)
        for id_batch in range(nd_batches):
            Xdb = Xdbs[id_batch]
            Ydb = Ydbs[id_batch]
            Ymbs = [model(Xdb) for model in models]
            Embs = [
                Ymbs[i] - tf.reshape(Ydb[:, i], (Ydb.shape[0], 1))
                for i in range(p.n_var)
            ]
            sum_E2s = np.array([tf.reduce_sum(E**2) for E in Embs])
            sum_E2 += sum_E2s
            # End of data batches.
        Ldats = np.array([np.sqrt(E2/n_data) for E2 in sum_E2])
        Ldat = np.sum(Ldats)
        losses_dat[epoch] = Ldats
        losses_dat[epoch][-1] = Ldat

        # Part 3c: Compute total loss.
        losses[epoch] = w_res*Lress + w_data*Ldats
        L = losses[epoch][-1] = w_res*Lres + w_data*Ldat
        if verbose:
            print(f"epoch = {epoch}, (Lres, Ldat, L) = "
                  f"({Lres:.6E}, {Ldat:.6E}, {L:.6E})")

        # --------------------------------------------------------------------

        # Save the trained models.
        if save_model > 0 and epoch % save_model == 0:
            common.save_models(
                models, output_dir, epoch, p.dependent_variable_names, multi)

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
        # print(f"Final value of loss function: {L}")

    # ------------------------------------------------------------------------

    # Save the final trained models and descriptions.
    if save_model != 0:
        common.save_models(
            models, output_dir, epoch, p.dependent_variable_names, multi)

    # Save the loss histories.
    path = os.path.join(output_dir, 'L_res.dat')
    np.savetxt(path, losses_res)
    path = os.path.join(output_dir, 'L_dat.dat')
    np.savetxt(path, losses_dat)
    path = os.path.join(output_dir, 'L.dat')
    np.savetxt(path, losses)


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
