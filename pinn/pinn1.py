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
DESCRIPTION = 'Solve a set of coupled 1st-order PDE using the PINN method.'

# Program defaults

# Default activation function to use in hidden nodes
DEFAULT_ACTIVATION = 'sigmoid'

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
DEFAULT_PRECISION = 'float32'

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
        '--activation', '-a', default=DEFAULT_ACTIVATION,
        help='Specify activation function (default: %(default)s)'
    )
    parser.add_argument(
        '--debug', '-d', action='store_true',
        help='Print debugging output (default: %(default)s)'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=DEFAULT_LEARNING_RATE,
        help='Initial learning rate for training (default: %(default)s)'
    )
    parser.add_argument(
        '--load_model', default=None,
        help='Path to directory containing models to load (default:'
             ' %(default)s)'
    )
    parser.add_argument(
        '--max_epochs', type=int, default=DEFAULT_MAX_EPOCHS,
        help='Maximum number of training epochs (default: %(default)s)'
    )
    parser.add_argument(
        '--multi', action='store_true',
        help='Use a single multi-output network (default: %(default)s)'
    )
    parser.add_argument(
        '--n_hid', type=int, default=DEFAULT_N_HID,
        help='Number of hidden nodes per layer (default: %(default)s)'
    )
    parser.add_argument(
        '--n_layers', type=int, default=DEFAULT_N_LAYERS,
        help='Number of hidden layers (default: %(default)s)'
    )
    parser.add_argument(
        '--nogpu', action='store_true',
        help='Disable TensorFlow use of GPU(s) (default: %(default)s)'
    )
    parser.add_argument(
        '--precision', type=str, default=DEFAULT_PRECISION,
        help='Precision to use in TensorFlow solution (default: %(default)s)'
    )
    parser.add_argument(
        '--save_model', type=int, default=DEFAULT_SAVE_MODEL,
        help='Save interval (epochs) for trained model (0 = do not save, '
        '-1 = save at end, n > 0 = save every n epochs) (default: %(default)s)'
    )
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_SEED,
        help='Seed for random number generator (default: %(default)s)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Print verbose output (default: %(default)s).'
    )
    parser.add_argument(
        '--w_data', '-w', type=float, default=DEFAULT_W_DATA,
        help='Normalized weight for data loss function '
             '(default: %(default)s).'
    )
    parser.add_argument(
        'problem_path',
        help='Path to problem description file (in python)'
    )
    parser.add_argument(
        'data_path',
        help='Path to problem data (IC, BC, etc.) file (default: %(default)s)'
    )
    parser.add_argument(
        'training_path',
        help='Path to training points file'
    )
    return parser


def main():
    """Begin main program."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug: print(f'args = {args}')
    activation = args.activation
    debug = args.debug
    learning_rate = args.learning_rate
    load_model = args.load_model
    max_epochs = args.max_epochs
    multi = args.multi
    H = args.n_hid
    n_layers = args.n_layers
    nogpu = args.nogpu
    precision = args.precision
    save_model = args.save_model
    seed = args.seed
    verbose = args.verbose
    w_data = args.w_data
    problem_path = args.problem_path
    data_path = args.data_path
    training_path = args.training_path

    # -------------------------------------------------------------------------

    # Configure TensorFlow.

    # If requested, disable TensorFlow use of GPU.
    if nogpu:
        if verbose:
            print('Disabling TensorFlow use of GPU.')
        common.disable_gpus()

    # Set the backend TensorFlow precision.
    if verbose:
        print(f'Setting TensorFlow precision to {precision}.')
    tf.keras.backend.set_floatx(precision)

    # Set the random number seed for reproducibility.
    if verbose:
        print(f'Seeding random number generator with {seed}.')
    tf.random.set_seed(seed)

    # -------------------------------------------------------------------------

    # Read the problem description.

    # Import the problem to solve.
    if verbose:
        print(f'Importing python module for problem {problem_path}.')
    p = common.import_problem(problem_path)
    if debug: print(f'{p = }')

    # Set up the output directory under the current directory.
    # The name of the output directory is the name of the problem python
    # module, with '-pinn1' appended to the end of the name.
    output_dir = os.path.join('.', f'{p.__name__}-pinn1')
    if debug: print(f'{output_dir = }')
    os.mkdir(output_dir)

    # Record system information, model parameters, and problem definition,
    # data, and training grid.
    if verbose:
        print('Recording system information, model hyperparameters, and '
              'problem definition and data.')
    common.save_system_information(output_dir)
    common.save_arguments(args, output_dir)
    shutil.copy(problem_path, output_dir)
    shutil.copy(data_path, output_dir)
    shutil.copy(training_path, output_dir)

    # -------------------------------------------------------------------------

    # Read the training points.

    # These are just coordinate tuples, one per line, space-delimited.
    if verbose:
        print(f'Reading training points from {training_path}.')
    # X_train is np.ndarray of shape (n_train, p.n_dim) OR (n_train,) for 1D.
    X_train = np.loadtxt(training_path, dtype=precision)
    if debug: print(f'{X_train = }')

    # If the data shape is 1-D (only one dimension), reshape to 2-D,
    # (n_train, 1) to make compatible with later TensorFlow calls, which
    # expect a 2D Tensor.
    if len(X_train.shape) == 1:
        if verbose:
            print('Training points are 1-D, reshaping to 2-D.')
        X_train = X_train.reshape(X_train.shape[0], 1)
        if debug: print(f'Reshaped X_train = {X_train}')

    # Save a copy of the training data in the output directory.
    shutil.copy(training_path, os.path.join(output_dir, 'X_train.dat'))

    # Count the training points.
    n_train = X_train.shape[0]
    if debug: print(f'{n_train = }')

    # -------------------------------------------------------------------------

    # Read the data points, which includes initial conditions, boundary
    # conditions, and any other data to be assimilated.

    # Each line contains a coordinate tuple, followed by a variables tuple,
    # containing the value of each variable at that location.
    if verbose:
        print(f'Reading training data from {data_path}.')
    # Shape is (n_data, p.n_dim + p.n_var)
    XY_data = np.loadtxt(data_path, dtype=precision)
    if debug: print(f'{XY_data = }')

    # If the data shape is 1-D (only one dimension), reshape to 2-D,
    # (n_train, 1) to make compatible with later TensorFlow calls, which
    # expect a 2D Tensor.
    if len(XY_data.shape) == 1:
        if verbose:
            print('Additional data is 1-D, reshaping to 2-D.')
        XY_data = XY_data.reshape(1, XY_data.shape[0])
        if debug: print(f'Reshaped XY_data = {XY_data}')

    # Save a copy of the additional data in the output directory.
    shutil.copy(data_path, os.path.join(output_dir, 'XY_data.dat'))

    # Get the count of training data points.
    n_data = XY_data.shape[0]
    if debug: print(f'{n_data = }')

    # Extract the *locations* of the supplied data points.
    # Shape is (n_data, p.n_dim)
    X_data = XY_data[:, :p.n_dim]
    if debug: print(f'{X_data = }')

    # Extract the *values* of the supplied data points.
    # Shape is (n_data, p.n_var)
    Y_data = XY_data[:, p.n_dim:]
    if debug: print(f'{Y_data = }')

    # -------------------------------------------------------------------------

    # Compute weights for residual and data loss functions.
    w_res = 1.0 - w_data
    if debug:
        print(f'{w_res = }')
        print(f'{w_data = }')

    # -------------------------------------------------------------------------

    # Create a model for each differential equation, unless 'multi' was
    # requested. If 'multi', create a single multi-output network.
    models = []
    if multi:
        if load_model:
            if verbose:
                print('Loading trained multi-output model.')
            raise TypeError(
                'Loading trained multi-output model not implemented!'
            )
        else:
            if verbose:
                print('Creating untrained multi-output model.')
            model = common.build_multi_output_model(n_layers, H, activation,
                                                    p.n_var)
            if debug: print(f'{model = }')
            models.append(model)
    else:
        if load_model:
            if verbose:
                print(f'Loading trained models from {load_model}.')
            for (i, v) in enumerate(p.dependent_variable_names):
                if verbose:
                    print(f'Loading model for {v}.')
                path = os.path.join(load_model, f'model_{v}')
                if debug: print(f'{path = }')
                model = tf.keras.models.load_model(path)
                if debug: print(f'{model = }')
                models.append(model)
        else:
            if verbose:
                print('Creating untrained models.')
            for (i, v) in enumerate(p.dependent_variable_names):
                if verbose:
                    print(f'Creating model for {v}.')
                model = common.build_model(n_layers, H, activation)
                if debug: print(f'{model = }')
                models.append(model)
        if debug: print(f'{models = }')

    # -------------------------------------------------------------------------

    # Create the optimizer to use for training.
    if verbose:
        print('Creating Adam optimizer.')
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    if debug: print(f'{optimizer = }')

    # -------------------------------------------------------------------------

    # Prepare inputs for TensorFlow.

    # Convert training points to tf.Variable.
    if verbose:
        print('Converting training points to TensorFlow Variable')
    X_train_tf = tf.Variable(X_train)
    if debug: print(f'{X_train_tf = }')

    # Convert data locations to tf.Variable.
    if verbose:
        print('Converting data locations to TensorFlow Variable')
    X_data_tf = tf.Variable(X_data)
    if debug: print(f'{X_data_tf = }')

    # Convert data values to tf.Variable.
    if verbose:
        print('Converting data values to TensorFlow Variable')
    Y_data_tf = tf.Variable(Y_data)
    if debug: print(f'{Y_data_tf = }')

    # -------------------------------------------------------------------------

    # Create loss histories as Python lists, so they can be easily updated.
    loss = {}
    for v in p.dependent_variable_names:
        loss[v] = {}
        loss[v]['residual'] = []
        loss[v]['data'] = []
        loss[v]['total'] = []
    loss['aggregate'] = {}
    loss['aggregate']['residual'] = []
    loss['aggregate']['data'] = []
    loss['aggregate']['total'] = []

    # -------------------------------------------------------------------------

    # Train the models.

    # Training involves presenting the training points and data points
    # together, a total of max_epochs times.

    # Record the training start time.
    t_start = datetime.datetime.now()
    if verbose:
        print(f'Training started at {t_start}.')

    # Main training loop
    for epoch in range(max_epochs):
        if debug: print(f'Starting epoch {epoch}.')

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
                if multi:
                    Y_multi = models[0](X_train_tf)
                    Y_train_model = [tf.reshape(Y_multi[:, i], (n_train, 1))
                                     for i in range(p.n_var)]
                else:
                    Y_train_model = [model(X_train_tf) for model in models]
                if debug: print(f'{Y_train_model = }')

                # Compute the network outputs at all data points. These
                # are the values of the dependent variables Y to use when
                # comparing to the supplied data.
                # Y_data_model is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list (one per model).
                # Each Tensor has shape (n_data, 1).
                Y_data_model = []
                if multi:
                    Y_multi_data = models[0](X_data_tf)
                    Y_data_model = [tf.reshape(Y_multi_data[:, i], (n_data, 1))
                                     for i in range(p.n_var)]
                else:
                    Y_data_model = [model(X_data_tf) for model in models]
                if debug: print(f'{Y_data_model = }')

            # Compute the gradients of the network outputs wrt inputs for
            # the training points. These are the values of the partial
            # derivatives dY/dX to use in the differential equations G.
            # dY_dX_train_model is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape (n_train, p.n_dim).
            dY_dX_train_model = [tape1.gradient(Y, X_train_tf)
                                 for Y in Y_train_model]
            if debug: print(f'{dY_dX_train_model = }')

            # Compute the values of the differential equations at all
            # training points.
            # G_train_model is a list of Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape (n_train, 1).
            G_train_model = [f(X_train_tf, Y_train_model, dY_dX_train_model)
                             for f in p.de]
            if debug: print(f'{G_train_model = }')

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
            if debug: print(f'{L_res_per_model = }')

            # Compute the errors in the predicted values at the data points.
            # E_data_per_model is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_data, 1).
            E_data_per_model = [
                Y_data_model[i] - tf.reshape(Y_data_tf[:, i], (n_data, 1))
                for i in range(p.n_var)
            ]
            if debug: print(f'{E_data_per_model = }')

            # Compute the loss functions for the data points for each
            # model.
            # L_data_per_model is a list of Tensor objects.
            # There are p.n_var Tensors in the list (one per model).
            # Each Tensor has shape () (scalar).
            L_data_per_model = [
                tf.math.sqrt(tf.reduce_sum(E**2)/n_data)
                for E in E_data_per_model
            ]
            if debug: print(f'{L_data_per_model = }')

            # Compute the weighted aggregate loss function per model.
            L_per_model = [
                w_res*L1 + w_data*L2 for (L1, L2)
                in zip(L_res_per_model, L_data_per_model)
            ]
            if debug: print(f'{L_per_model = }')

            # Compute the aggregated residual loss function.
            L_res = tf.math.reduce_sum(L_res_per_model)
            if debug: print(f'{L_res = }')

            # Compute the aggregated data loss function.
            L_data = tf.math.reduce_sum(L_data_per_model)
            if debug: print(f'{L_data = }')

            # Compute the weighted aggregate loss function.
            L = w_res*L_res + w_data*L_data
            if debug: print(f'{L = }')

            # Abort if a NaN was produced.
            if np.isnan(L):
                print(f'Aborting at epoch {epoch}: loss is NaN!')
                break

            # Save the losses for this epoch.
            for (i, v) in enumerate(p.dependent_variable_names):
                loss[v]['residual'].append(L_res_per_model[i].numpy())
                loss[v]['data'].append(L_data_per_model[i].numpy())
                loss[v]['total'].append(L_per_model[i].numpy())
            loss['aggregate']['residual'].append(L_res.numpy())
            loss['aggregate']['data'].append(L_data.numpy())
            loss['aggregate']['total'].append(L.numpy())
            if debug: print(f'{loss = }')

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
        if debug: print(f'{pgrad = }')

        # Update the parameters for this epoch.
        for (g, m) in zip(pgrad, models):
            optimizer.apply_gradients(zip(g, m.trainable_variables))

        # --------------------------------------------------------------------

        if verbose:
            print(f'epoch = {epoch}, (L_res, L_data, L) = '
                  f'({L_res:6e}, {L_data:6e}, {L:6e})')

        # Save the trained models.
        if save_model > 0 and epoch % save_model == 0:
            if multi:
                path = os.path.join(
                    output_dir, 'models', f'{epoch:06d}', 'model_multi'
                )
                model.save(path)
            else:
                for (i, model) in enumerate(models):
                    path = os.path.join(
                        output_dir, 'models', f'{epoch:06d}',
                        f'model_{p.dependent_variable_names[i]}'
                    )
                    model.save(path)

        if debug: print(f'Ending epoch {epoch}.')

    # Count the last epoch.
    n_epochs = epoch + 1
    if debug: print(f'{n_epochs = }')

    # Record the training end time.
    t_stop = datetime.datetime.now()
    t_elapsed = t_stop - t_start
    if verbose:
        print(f'Training stopped at {t_stop}.')
        print(f'Total training time: {t_elapsed.total_seconds()} seconds')
        print(f'Epochs: {n_epochs}')
        print(f'Final value of loss function: {L}')

    # Save the final trained models and descriptions.
    if save_model != 0:
        if multi:
            path = os.path.join(
                output_dir, 'models', f'{epoch:06d}', 'model_multi'
            )
            model.save(path)
            path = os.path.join(output_dir, 'models', 'model_multi.txt')
            old_stdout = sys.stdout
            with open(path, 'w') as f:
                sys.stdout = f
                model.summary()
            sys.stdout = old_stdout
        else:
            for (i, model) in enumerate(models):
                path = os.path.join(
                    output_dir, 'models', f'{epoch:06d}',
                    f'model_{p.dependent_variable_names[i]}'
                )
                model.save(path)
                variable_name = p.dependent_variable_names[i]
                path = os.path.join(output_dir, 'models',
                                    f'model_{variable_name}.txt')
                old_stdout = sys.stdout
                with open(path, 'w') as f:
                    sys.stdout = f
                    model.summary()
                sys.stdout = old_stdout

    # Save the loss histories.
    for (i, v) in enumerate(p.dependent_variable_names):
        np.savetxt(
            os.path.join(output_dir, f'L_res_{v}.dat'), loss[v]['residual']
        )
        np.savetxt(
            os.path.join(output_dir, f'L_data_{v}.dat'), loss[v]['data']
        )
        np.savetxt(
            os.path.join(output_dir, f'L_{v}.dat'), loss[v]['total']
        )
    np.savetxt(
        os.path.join(output_dir, 'L_res.dat'), loss['aggregate']['residual']
    )
    np.savetxt(
        os.path.join(output_dir, 'L_data.dat'), loss['aggregate']['data']
    )
    np.savetxt(
        os.path.join(output_dir, 'L.dat'), loss['aggregate']['total']
    )


if __name__ == '__main__':
    """Begin main program."""
    main()
