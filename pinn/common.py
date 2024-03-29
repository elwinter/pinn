"""Common code for pinn package.

This module provides a set of standard functions used by all of the programs
in the pinn package.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.
import datetime
import glob
import importlib
import os
import platform
import shutil
import sys

# Import 3rd-party modules.
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


# Module constants

# Name of file to hold the system information report.
system_information_file = "system_information.txt"

# Name of file to hold the network hyperparameters, as an importable Python
# module.
hyperparameter_file = "hyperparameters.py"

# Initial parameter ranges
w0_range = [-0.1, 0.1]  # Hidden layer weights
u0_range = [-0.1, 0.1]  # Hidden layer biases
v0_range = [-0.1, 0.1]  # Output layer weights


def build_model(n_layers, n_hidden, activation):
    """Build a multi-layer neural network model.

    Build a fully-connected, multi-layer neural network with single output.
    Each layer will have H hidden nodes. Each hidden node has weights and
    a bias, and uses the specified activation function.

    The number of inputs is determined when the network is first used.

    Parameters
    ----------
    n_layers : int
        Number of hidden layers to create.
    n_hidden : int
        Number of nodes to use in each hidden layer.
    activation : str
        Name of activation function (from TensorFlow) to use.

    Returns
    -------
    model : tf.keras.Sequential
        The neural network.
    """
    layers = []
    for _ in range(n_layers):
        hidden_layer = tf.keras.layers.Dense(
            units=n_hidden, use_bias=True,
            activation=tf.keras.activations.deserialize(activation),
            kernel_initializer=tf.keras.initializers.RandomUniform(*w0_range),
            bias_initializer=tf.keras.initializers.RandomUniform(*u0_range)
        )
        layers.append(hidden_layer)
    output_layer = tf.keras.layers.Dense(
        units=1,
        activation=tf.keras.activations.linear,
        kernel_initializer=tf.keras.initializers.RandomUniform(*v0_range),
        use_bias=False,
    )
    layers.append(output_layer)
    model = tf.keras.Sequential(layers)
    return model


def save_hyperparameters(args, output_dir):
    """Save the neural network hyperparameters.

    Print a record of the hyperparameters of the neural network in the
    specified directory, as an importable python module.

    Parameters
    ----------
    args : dict
        Dictionary of command-line arguments.
    output_dir : str
        Path to directory to contain the report.

    Returns
    -------
    path : str
        Path to hyperparameter file.

    Raises
    ------
    None
    """
    path = os.path.join(output_dir, hyperparameter_file)
    with open(path, "w") as f:
        f.write(f"activation = {repr(args.activation)}\n")
        if "batch_size" in args:
            f.write(f"batch_size = {repr(args.batch_size)}\n")
        f.write(f"data = {repr(args.data)}\n")
        f.write(f"learning_rate = {repr(args.learning_rate)}\n")
        f.write(f"max_epochs = {repr(args.max_epochs)}\n")
        f.write(f"n_hid = {repr(args.n_hid)}\n")
        f.write(f"n_layers = {repr(args.n_layers)}\n")
        f.write(f"nogpu = {repr(args.nogpu)}\n")
        f.write(f"precision = {repr(args.precision)}\n")
        f.write(f"random_seed = {repr(args.seed)}\n")
        f.write(f"validation = {repr(args.validation)}\n")
        if "w_data" in args:
            f.write(f"w_data = {repr(args.w_data)}\n")
        f.write(f"problem_path = {repr(args.problem_path)}\n")
        if "training_points" in args:
            f.write(f"training_points = {repr(args.training_points)}\n")
    return path


def save_problem_definition(problem, output_dir):
    """Save the problem definition for the run.

    Copy the problem definition file to the output directory.

    Parameters
    ----------
    problem : module
        Imported module object for problem definition.
    output_dir : str
        Path to directory to contain the copy of the problem definition file.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Copy the problem definition file to the output directory.
    shutil.copy(problem.__file__, output_dir)


def save_system_information(output_dir):
    """Save a summary of system characteristics.

    Save a summary of the host system in the specified directory.

    Parameters
    ----------
    output_dir : str
        Path to directory to contain the report.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    path = os.path.join(output_dir, system_information_file)
    with open(path, "w") as f:
        f.write("System report:\n")
        f.write(f"Start time: {datetime.datetime.now()}\n")
        f.write(f"Host name: {platform.node()}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"uname: {' '.join(platform.uname())}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Python build: {' '.join(platform.python_build())}\n")
        f.write(f"Python compiler: {platform.python_compiler()}\n")
        f.write(f"Python implementation: {platform.python_implementation()}\n")
        f.write(f"NumPy version: {np.__version__}\n")
        f.write(f"TensorFlow version: {tf.__version__}\n")
        f.write("Available TensorFlow devices: "
                f"{device_lib.list_local_devices()}\n")


def find_last_epoch(results_path):
    """Find the last epoch for a model in the results directory.

    Find the last epoch for a model in the results directory.

    Parameters
    ----------
    results_path : str
        Path to results directory.

    Returns
    -------
    last_epoch : int
        Number for last epoch found in results directory.
    """
    # Save the current directory.
    original_directory = os.getcwd()

    # Construct the path to the saved models.
    models_directory = os.path.join(results_path, "models")

    # Move to the saved models directory.
    os.chdir(models_directory)

    # Make a list of all subdirectories with names starting with digits.
    # These digits represent epoch numbers at which the models were saved.
    epoch_directories = glob.glob("[0-9]*")

    # Return to the original directory.
    os.chdir(original_directory)

    # Find the largest epoch number.
    epochs = [int(s) for s in epoch_directories]
    last_epoch = max(epochs)

    # Return the largest epoch number.
    return last_epoch


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
    """
    problem_name = os.path.splitext(os.path.split(problem_path)[-1])[-2]
    spec = importlib.util.spec_from_file_location(problem_name, problem_path)
    p = importlib.util.module_from_spec(spec)
    sys.modules[problem_name] = p
    spec.loader.exec_module(p)
    return p


if __name__ == '__main__':
    pass
