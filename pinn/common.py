"""Common code for pinn package.

This module provides a set of standard functions used by all of the programs
in the pinn package.

Notes on code:

* cproc = generic CompletedProcess object from subprocess,run()

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.
import argparse
import datetime
import glob
import importlib
import os
import platform
import subprocess
import sys

# Import supplemental modules.
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


# ----------------------------------------------------------------------------

# Command-line utilities


def create_minimal_command_line_argument_parser(description=''):
    """Create a minimal command-line argument parser.

    Create a minimal command-line argument parser.

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
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        '--debug', '-d', action='store_true',
        help="Print debugging output (default: %(default)s)"
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help="Print verbose output (default: %(default)s)."
    )
    return parser


# Defaults for neural network command-line options
DEFAULT_ACTIVATION = 'sigmoid'
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_N_LAYERS = 1
DEFAULT_N_HID = 10
DEFAULT_MAX_EPOCHS = 100
DEFAULT_TENSORFLOW_PRECISION = 'float32'
DEFAULT_SEED = 0

# Default model save interval (in epochs) for saving the model.
# 0 = do not save model
# -1 = only save at end
# n > 0: Save after every n epochs.
DEFAULT_SAVE_MODEL = -1


def create_neural_network_command_line_argument_parser(description=''):
    """Create a command-line argument parser for neural network code.

    Create a command-line argument parser for neural network code.

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
    parser = create_minimal_command_line_argument_parser(description)
    parser.add_argument(
        '--n_layers', type=int, default=DEFAULT_N_LAYERS,
        help="Number of hidden layers (default: %(default)s)"
    )
    parser.add_argument(
        '--n_hid', type=int, default=DEFAULT_N_HID,
        help="Number of hidden nodes per layer (default: %(default)s)"
    )
    parser.add_argument(
        '--activation', '-a', default=DEFAULT_ACTIVATION,
        help="Specify activation function (default: %(default)s)"
    )
    parser.add_argument(
        '--learning_rate', type=float, default=DEFAULT_LEARNING_RATE,
        help="Initial learning rate for training (default: %(default)s)"
    )
    parser.add_argument(
        '--max_epochs', type=int, default=DEFAULT_MAX_EPOCHS,
        help="Maximum number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        '--precision', type=str, default=DEFAULT_TENSORFLOW_PRECISION,
        help="Precision to use in TensorFlow solution (default: %(default)s)"
    )
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_SEED,
        help="Seed for random number generator (default: %(default)s)"
    )
    parser.add_argument(
        '--save_model', type=int, default=DEFAULT_SAVE_MODEL,
        help="Save interval (epochs) for trained model (0 = do not save, "
        "-1 = save at end, n > 0 = save every n epochs) (default: %(default)s)"
    )
    parser.add_argument(
        '--nogpu', action='store_true',
        help="Disable TensorFlow use of GPU(s) (default: %(default)s)"
    )
    parser.add_argument(
        'problem_path',
        help="Path to problem description file (in python)"
    )
    parser.add_argument(
        'training_path',
        help='Path to training points file (as a PINN grid file)'
    )
    return parser


# ----------------------------------------------------------------------------

# General program utilities

# Name of file to hold the system information report.
SYSTEM_INFORMATION_FILE = 'system_information.txt'

# Name of file to hold the program arguments, as an importable Python
# module.
ARGUMENTS_FILE = 'arguments.py'


def save_arguments(args_ns, output_dir):
    """Save the program arguments.

    Save a record of the program arguments in the specified directory, as an
    importable python module. The arguments are saved to the file in sorted
    order.

    Parameters
    ----------
    args_ns : Namespace
        Namespace of command-line arguments
    output_dir : str
        Path to directory to contain the report

    Returns
    -------
    path : str
        Path to arguments file

    Raises
    ------
    None
    """
    # Convert argument Namespace to a dict.
    args = vars(args_ns)
    path = os.path.join(output_dir, ARGUMENTS_FILE)
    with open(path, 'w', encoding='utf-8') as f:
        for arg in sorted(args):
            f.write(f"{arg} = {repr(args[arg])}\n")
    return path


def save_system_information(output_dir):
    """Save a summary of system characteristics.

    Save a summary of the host system in the specified directory.

    Parameters
    ----------
    output_dir : str
        Path to directory to contain the report.

    Returns
    -------
    path : str
        Path to system information file

    Raises
    ------
    None
    """
    path = os.path.join(output_dir, SYSTEM_INFORMATION_FILE)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('System information:\n')
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
        f.write('Available TensorFlow devices: '
                f"{device_lib.list_local_devices()}\n")
        f.write(f"conda environment: {os.environ['CONDA_DEFAULT_ENV']}\n")
        f.write(f"Git branch: {get_git_branch()}\n")
        f.write(f"Latest git hash: {get_git_hash()}\n")
    return path


# ----------------------------------------------------------------------------

# git utilities


def get_git_branch():
    """Get the current git branch.

    Get the current git branch.

    Parameters
    ----------
    None

    Returns
    -------
    git_branch : str
        Name of current gir branch

    Raises
    ------
    subprocess.CalledProcessError
        If unable to determine git branch
    """
    cwd = os.getcwd()
    os.chdir(os.environ['PINN_ROOT'])
    cmd = 'git branch'
    cproc = subprocess.run(cmd, shell=True, check=True, text=True,
                           capture_output=True)
    lines = cproc.stdout.splitlines()
    git_branch = None
    for line in lines:
        if line.startswith('*'):
            git_branch = line[2:]
    if git_branch is None:
        raise subprocess.CalledProcessError('Unable to determine git branch!')
    os.chdir(cwd)
    return git_branch


def get_git_hash():
    """Get the current git hash.

    Get the current git hash.

    Parameters
    ----------
    None

    Returns
    -------
    git_hash : str
        Hash for current commit

    Raises
    ------
    subprocess.CalledProcessError
        If unable to determine git hash
    """
    cwd = os.getcwd()
    os.chdir(os.environ['PINN_ROOT'])
    cmd = 'git rev-parse HEAD'
    cproc = subprocess.run(cmd, shell=True, check=True, text=True,
                           capture_output=True)
    git_hash = str(cproc.stdout.rstrip())  # Originally bytes
    os.chdir(cwd)
    return git_hash


# ----------------------------------------------------------------------------

# Tensorflow utilities


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
    tf.config.set_visible_devices([], 'GPU')

    # Make sure the GPU were disabled.
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'


# ----------------------------------------------------------------------------

# Neural network utilities

# Initial parameter ranges
W0_RANGE = [-0.1, 0.1]  # Hidden layer weights
U0_RANGE = [-0.1, 0.1]  # Hidden layer biases
V0_RANGE = [-0.1, 0.1]  # Output layer weights


def build_model(n_layers, n_hidden, activation):
    """Build a multi-layer neural network model.

    Build a fully-connected, multi-layer neural network with single output.
    Each layer will have H hidden nodes. Each hidden node has weights and
    a bias, and uses the specified activation function. The output layer
    does not use a bias.

    Weights and biases are initialized with a uniformed rndom distribution
    in the ranges defined in W0_RANGE, U0_RANGE, and V0_RANGE.

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

    Raises
    ------
    None
    """
    layers = []
    for _ in range(n_layers):
        hidden_layer = tf.keras.layers.Dense(
            units=n_hidden, use_bias=True,
            activation=tf.keras.activations.deserialize(activation),
            kernel_initializer=tf.keras.initializers.RandomUniform(*W0_RANGE),
            bias_initializer=tf.keras.initializers.RandomUniform(*U0_RANGE)
        )
        layers.append(hidden_layer)
    output_layer = tf.keras.layers.Dense(
        units=1,
        activation=tf.keras.activations.linear,
        kernel_initializer=tf.keras.initializers.RandomUniform(*V0_RANGE),
        use_bias=False,
    )
    layers.append(output_layer)
    model = tf.keras.Sequential(layers)
    return model


# def build_multi_output_model(n_layers, n_hidden, activation, n_out):
#     """Build a multi-output, multi-layer neural network model.

#     Build a fully-connected, multi-layer neural network with multiple outputs.
#     Each layer will have H hidden nodes. Each hidden node has weights and
#     a bias, and uses the specified activation function.

#     The number of inputs is determined when the network is first used.

#     Parameters
#     ----------
#     n_layers : int
#         Number of hidden layers to create.
#     n_hidden : int
#         Number of nodes to use in each hidden layer.
#     activation : str
#         Name of activation function (from TensorFlow) to use.
#     n_out : int
#         Number of network outputs

#     Returns
#     -------
#     model : tf.keras.Sequential
#         The neural network.
#     """
#     layers = []
#     for _ in range(n_layers):
#         hidden_layer = tf.keras.layers.Dense(
#             units=n_hidden, use_bias=True,
#             activation=tf.keras.activations.deserialize(activation),
#             kernel_initializer=tf.keras.initializers.RandomUniform(*w0_range),
#             bias_initializer=tf.keras.initializers.RandomUniform(*u0_range)
#         )
#         layers.append(hidden_layer)
#     output_layer = tf.keras.layers.Dense(
#         units=n_out,
#         activation=tf.keras.activations.linear,
#         kernel_initializer=tf.keras.initializers.RandomUniform(*v0_range),
#         use_bias=False,
#     )
#     layers.append(output_layer)
#     model = tf.keras.Sequential(layers)
#     return model


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

    Raises
    ------
    None
    """
    # Save the current directory.
    original_directory = os.getcwd()

    # Construct the path to the saved models.
    models_directory = os.path.join(results_path, 'models')

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

# ----------------------------------------------------------------------------

# General PINN utilities


def import_problem(problem_path):
    """Import the Python file which defines the problem to solve.

    Import the Python file which defines the problem to solve.

    Note that the absolute path is required by module_from_spec().

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
    abspath = os.path.abspath(problem_path)
    problem_name = os.path.splitext(os.path.split(abspath)[-1])[-2]
    spec = importlib.util.spec_from_file_location(problem_name, abspath)
    p = importlib.util.module_from_spec(spec)
    sys.modules[problem_name] = p
    spec.loader.exec_module(p)
    return p


# def read_grid_description(data_file):
#     """Read grid description from a data file.

#     Read grid description from a data file. If the data is random, then return
#     None.

#     Parameters
#     ----------
#     data_file : str
#         Path to training data file

#     Returns
#     -------
#     xg : list of list of float
#         List of pairs of (min, max) for each grid dimension
#     ng : list of int
#         Number of grid points in each dimension

#     Raises
#     ------
#     None
#     """
#     # Read the grid description. Ignore if not a grid.
#     xg = None
#     ng = None
#     with open(data_file, "r") as f:
#         line = f.readline()
#         if line.startswith("# GRID"):
#             line = f.readline().rstrip()
#             line = line[2:]
#             f = line.split(" ")
#             xmin = f[::3]
#             xmax = f[1::3]
#             xn = f[2::3]
#             xg = []
#             ng = []
#             for (min, max, n) in zip(xmin, xmax, xn):
#                 xg.append([None, None])
#                 xg[-1][0] = float(min)
#                 xg[-1][1] = float(max)
#                 ng.append(int(n))
#         else:
#             pass
#     return xg, ng


def read_grid_file(path):
    """Read grid description and data from a file.

    Read the grid description and data from a file. The file is assumed to
    contain a grid header, followed list of nrows points. Each point
    definines a grid location (x0, x1, ...) and zero or more values defined
    at that location (y0, y1, ...).

    Parameters
    ----------
    path : str
        Path to grid file

    Returns
    -------
    column_descriptions : list of ncols tuples, each (str, float, float, int)
        List of (varname, min, max, n) for each grid dimension.
    data : np.ndarray, shape (nrows, ncols)
        Numpy array of data in file

    Raises
    ------
    AssertionError
        If this is not a grid file
    """
    # Read the grid description.
    column_descriptions = []
    COMMENT_PREFIX = '# '
    COMMENT_PREFIX_LEN = len(COMMENT_PREFIX)
    GRID_HEADER_LINE = f"{COMMENT_PREFIX}GRID"
    with open(path, 'r', encoding='utf-8') as f:

        # Make sure this is a grid file.
        line = f.readline().rstrip()
        assert line == GRID_HEADER_LINE

        # Read the names of the independent variables (dimensions) which
        # define the grid points.
        line = f.readline().rstrip()
        assert line.startswith(COMMENT_PREFIX)
        dim_names_str = line[COMMENT_PREFIX_LEN:]
        dim_names = dim_names_str.split(' ')
        n_dim = len(dim_names)
        assert n_dim > 0

        # Read the grid description.
        line = f.readline().rstrip()
        assert line.startswith(COMMENT_PREFIX)
        description_str = line[COMMENT_PREFIX_LEN:]
        fields = description_str.split(' ')
        xmin = [float(f) for f in fields[::3]]
        xmax = [float(f) for f in fields[1::3]]
        nx = [int(f) for f in fields[2::3]]

        # Read the column names.
        line = f.readline().rstrip()
        assert line.startswith(COMMENT_PREFIX)
        column_names_str = line[COMMENT_PREFIX_LEN:]
        column_names = column_names_str.split(' ')
        n_cols = len(column_names)
        n_var = n_cols - n_dim
        for i in range(n_dim):
            assert dim_names[i] == column_names[i]
        pad = [None]*n_var
        column_descriptions = [
            (s1, f1, f2, i1)
            for (s1, f1, f2, i1) in
            zip(column_names, xmin + pad, xmax + pad, nx + pad)
        ]

    # Now load the data table.
    data = np.loadtxt(path)

    # Return the grid description and data.
    return column_descriptions, data


if __name__ == '__main__':
    pass
