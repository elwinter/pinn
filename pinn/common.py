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
import shutil
import subprocess
import sys

# Import supplemental modules.
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


# Module constants

# Name of file to hold the program arguments, as an importable Python
# module.
ARGUMENTS_FILE = "arguments.py"

# Name of file to hold the system information report.
SYSTEM_INFORMATION_FILE = "system_information.txt"

# Initial model parameter ranges
W0_RANGE = [-0.1, 0.1]  # Hidden layer weights
U0_RANGE = [-0.1, 0.1]  # Hidden layer biases
V0_RANGE = [-0.1, 0.1]  # Output layer weights


# ----------------------------------------------------------------------------


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


def build_multi_output_model(n_layers, n_hidden, activation, n_out):
    """Build a multi-output, multi-layer neural network model.

    Build a fully-connected, multi-layer neural network with multiple outputs.
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
    n_out : int
        Number of network outputs

    Returns
    -------
    model : tf.keras.Sequential
        The neural network.

    Raises:
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
        units=n_out,
        activation=tf.keras.activations.linear,
        kernel_initializer=tf.keras.initializers.RandomUniform(*V0_RANGE),
        use_bias=False,
    )
    layers.append(output_layer)
    model = tf.keras.Sequential(layers)
    return model


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
    # If requested, disable TensorFlow use of GPU.
    if args["nogpu"]:
        disable_gpus()

    # Set the backend TensorFlow precision.
    tf.keras.backend.set_floatx(args["precision"])

    # Set the random number seed for reproducibility.
    tf.random.set_seed(args["seed"])


def create_batches(X_train: np.ndarray, batch_size: int = -1):
    """Split the data into batches of tf.Variable.

    Split the data into batches of tf.Variable.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, p.n_dim)
        Array of training points.
    batch_size : int, default -1
        Size of each batch. If -1, use a single batch.

    Returns
    -------
    batches : list of tf.Variable
        Training points split into batches.

    Raises
    ------
    None
    """
    batches = []
    if batch_size == -1:
        tfv = tf.Variable(X_train)
        batches.append(tfv)
    else:
        n_train = X_train.shape[0]
        n_batches = int(np.ceil(n_train/batch_size))
        for ib in range(n_batches):
            i_start = ib*batch_size
            i_end = (ib + 1)*batch_size
            i_end = min(i_end, n_train)
            tfv = tf.Variable(X_train[i_start:i_end])
            batches.append(tfv)

    # Return the list of batches.
    return batches


def create_minimal_command_line_argument_parser(description=""):
    """Create a minimal command-line argument parser.

    Create a minimal command-line argument parser. After creation, arguments
    may be added to this parser.

    Parameters
    ----------
    description: str, default ""
        Description string for script.

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
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    return parser


def create_models(p, args: dict):
    """Create one untrained PINN model per problem dependent variable.

    Create one untrained PINN model per problem dependent variable.

    Parameters
    ----------
    p : Python module object
        Imported module for the problem definition.
    args : dict
        Dictionary of command-line options.

    Returns
    -------
    models : list of keras.src.engine.sequential.Sequential, len p.n_var
        The new models.

    Raises
    ------
    None
    """
    models = None
    if args["multi"]:
        models = [
            build_multi_output_model(
                args["n_layers"], args["n_hid"], args["activation"], p.n_var
            )
        ]
    else:
        models = [
            build_model(args["n_layers"], args["n_hid"], args["activation"])
            for _ in p.dependent_variable_names
        ]

    # Return the models.
    return models


def create_neural_network_command_line_argument_parser(description=""):
    """Create a basic command-line argument parser for neural network code.

    Create a basic command-line argument parser for neural network code.

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
    # Create the minimal command-line parser.
    parser = create_minimal_command_line_argument_parser(description)

    # Add neural network-specific arguments.
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
        "--learning_rate", type=float, default=0.01,
        help="Initial learning rate for training (default: %(default)s)"
    )
    parser.add_argument(
        "--load_model", default=None,
        help="Path to directory containing models to load (default:"
             " %(default)s)"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10,
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
    return parser


def create_optimizer(learning_rate: float = 0.01):
    """Create the training optimizer.

    Create the training optimizer.

    Parameters
    ----------
    learning_rate : float, default 0.01
        Initial learning rate for optimizer.

    Returns
    -------
    optimizer : keras.src.optimizers.legacy.adam.Adam
        The optimizer to use for training.

    Raises
    ------
    None
    """
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

    # Return the optimizer.
    return optimizer


def create_output_directory(p, s: str = "", clobber=False):
    """Create the output directory for this problem.

    Create the output directory for this problem. The name of the output
    directory is the name of the problem python module, with s appended
    to the end of the name.

    Parameters
    ----------
    p : Python module object
        Imported module for the problem definition.
    s : str, default ""
        String to append to problem name in directory name.
    clobber : bool, default False
        If True, remove exiting directory of same name.

    Returns
    -------
    output_dir : str
        Name of output directory.

    Raises
    ------
    None
    """
    output_dir = os.path.join(".", f"{p.__name__}{s}")
    if os.path.isdir(output_dir) and clobber:
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    return output_dir


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
    AssertionError : If this code cannot disable all GPU.
    """
    # Disable all GPUS.
    tf.config.set_visible_devices([], "GPU")

    # Make sure the GPU were disabled.
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"


def get_git_branch():
    """Get the git branch for the current PINN code.

    Get the git branch for the current PINN code.

    Parameters
    ----------
    None

    Returns
    -------
    git_branch : str
        Name of current git branch

    Raises
    ------
    None
    """
    cwd = os.getcwd()
    os.chdir(os.environ["PINN_ROOT"])
    cmd = "git branch"
    cproc = subprocess.run(cmd, shell=True, check=True, text=True,
                           capture_output=True)
    lines = cproc.stdout.splitlines()
    git_branch = None
    for line in lines:
        if line.startswith("*"):
            git_branch = line[2:]
    os.chdir(cwd)
    return git_branch


def get_git_hash():
    """Get the git hash for the current PINN code.

    Get the git hash for the current PINN code.

    Parameters
    ----------
    None

    Returns
    -------
    git_hash : str
        Hash for current commit

    Raises
    ------
    None
    """
    cwd = os.getcwd()
    os.chdir(os.environ["PINN_ROOT"])
    cmd = "git rev-parse HEAD"
    cproc = subprocess.run(cmd, shell=True, check=True, text=True,
                           capture_output=True)
    git_hash = str(cproc.stdout.rstrip())  # Originally bytes
    os.chdir(cwd)
    return git_hash


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


def load_problem_data(path: str, precision: str = "float32"):
    """Load the problem data.

    Load the problem data, which specifies the function values at each
    point.

    Each line contains a tuple of coordinates (the independent variable values
    for a location), followed by a tuple of dependent variable values,
    containing the values of each dependent variable at that location.

    Parameters
    ----------
    path : str
        Path to problem data file to read.
    precision : str, default "float32"
        TensorFlow precision to use for data

    Returns
    -------
    XY_data : np.ndarray, shape(n_data, p.n_dim + p.n_var)
        Array of data points.

    Raises
    ------
    None
    """
    # Load the data.
    XY_data = np.loadtxt(path, dtype=precision)

    # If the data shape is 1-D (a single point), reshape to 2-D (1, n_data)
    # to make compatible with later TensorFlow calls, which expect a 2D
    # Tensor.
    if len(XY_data.shape) == 1:
        XY_data = XY_data.reshape(1, XY_data.shape[0])

    # Return the problem data.
    return XY_data


def load_trained_models(model_directory: str, model_names: list):
    """Load trained PINN models.

    Load trained PINN models.

    Parameters
    ----------
    model_directory : str
        Path to directory containing models
    model_names : list of str
        Names of models to load

    Returns
    -------
    models : list of keras.src.engine.sequential.Sequential
        The models read from storage.

    Raises
    ------
    None
    """
    models = []
    for model_name in model_names:
        path = os.path.join(model_directory, f"model_{model_name}")
        model = tf.keras.models.load_model(path)
        models.append(model)

    # Return the models.
    return models


def save_arguments(args: dict, output_dir: str = ""):
    """Save the program arguments.

    Save a record of the program arguments in the specified directory, as an
    importable python module. The arguments are saved to the file in sorted
    order.

    Parameters
    ----------
    args : dict
        Dictionary of command-line arguments
    output_dir : str, default ""
        Path to directory to contain the arguments file

    Returns
    -------
    path : str
        Path to arguments file

    Raises
    ------
    None
    """
    # Convert argument Namespace to a dict.
    path = os.path.join(output_dir, ARGUMENTS_FILE)
    with open(path, "w", encoding="utf-8") as f:
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
    with open(path, "w", encoding="utf-8") as f:
        f.write("System information:\n")
        f.write(f"Start time: {datetime.datetime.now()}\n")
        f.write(f"Host name: {platform.node()}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"uname: {' '.join(platform.uname())}\n")
        f.write(f"conda environment: {os.environ['CONDA_DEFAULT_ENV']}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Python build: {' '.join(platform.python_build())}\n")
        f.write(f"Python compiler: {platform.python_compiler()}\n")
        f.write(f"Python implementation: {platform.python_implementation()}\n")
        f.write(f"NumPy version: {np.__version__}\n")
        f.write(f"TensorFlow version: {tf.__version__}\n")
        f.write("Available TensorFlow devices: "
                f"{device_lib.list_local_devices()}\n")
        f.write(f"PINN git branch: {get_git_branch()}\n")
        f.write(f"PINN git hash: {get_git_hash()}\n")
    return path


# ----------------------------------------------------------------------------

# # General program utilities

# # Name of file to hold the system information report.
# SYSTEM_INFORMATION_FILE = 'system_information.txt'

# # Name of file to hold the program arguments, as an importable Python
# # module.
# ARGUMENTS_FILE = 'arguments.py'


# def save_arguments(args: dict, output_dir: str = ""):
#     """Save the program arguments.

#     Save a record of the program arguments in the specified directory, as an
#     importable python module. The arguments are saved to the file in sorted
#     order.

#     Parameters
#     ----------
#     args : dict
#         Dictionary of command-line arguments
#     output_dir : str, default ""
#         Path to directory to contain the arguments file

#     Returns
#     -------
#     path : str
#         Path to arguments file

#     Raises
#     ------
#     None
#     """
#     # Convert argument Namespace to a dict.
#     path = os.path.join(output_dir, ARGUMENTS_FILE)
#     with open(path, "w", encoding="utf-8") as f:
#         for arg in sorted(args):
#             f.write(f"{arg} = {repr(args[arg])}\n")
#     return path


# def save_system_information(output_dir):
#     """Save a summary of system characteristics.

#     Save a summary of the host system in the specified directory.

#     Parameters
#     ----------
#     output_dir : str
#         Path to directory to contain the report.

#     Returns
#     -------
#     path : str
#         Path to system information file

#     Raises
#     ------
#     None
#     """
#     path = os.path.join(output_dir, SYSTEM_INFORMATION_FILE)
#     with open(path, 'w', encoding='utf-8') as f:
#         f.write('System information:\n')
#         f.write(f"Start time: {datetime.datetime.now()}\n")
#         f.write(f"Host name: {platform.node()}\n")
#         f.write(f"Platform: {platform.platform()}\n")
#         f.write(f"uname: {' '.join(platform.uname())}\n")
#         f.write(f"Python version: {sys.version}\n")
#         f.write(f"Python build: {' '.join(platform.python_build())}\n")
#         f.write(f"Python compiler: {platform.python_compiler()}\n")
#         f.write(f"Python implementation: {platform.python_implementation()}\n")
#         f.write(f"NumPy version: {np.__version__}\n")
#         f.write(f"TensorFlow version: {tf.__version__}\n")
#         f.write('Available TensorFlow devices: '
#                 f"{device_lib.list_local_devices()}\n")
#         f.write(f"conda environment: {os.environ['CONDA_DEFAULT_ENV']}\n")
#         f.write(f"Git branch: {get_git_branch()}\n")
#         f.write(f"Latest git hash: {get_git_hash()}\n")
#     return path


# ----------------------------------------------------------------------------

# git utilities


# def get_git_branch():
#     """Get the current git branch.

#     Get the current git branch.

#     Parameters
#     ----------
#     None

#     Returns
#     -------
#     git_branch : str
#         Name of current gir branch

#     Raises
#     ------
#     subprocess.CalledProcessError
#         If unable to determine git branch
#     """
#     cwd = os.getcwd()
#     os.chdir(os.environ['PINN_ROOT'])
#     cmd = 'git branch'
#     cproc = subprocess.run(cmd, shell=True, check=True, text=True,
#                            capture_output=True)
#     lines = cproc.stdout.splitlines()
#     git_branch = None
#     for line in lines:
#         if line.startswith('*'):
#             git_branch = line[2:]
#     if git_branch is None:
#         raise subprocess.CalledProcessError('Unable to determine git branch!')
#     os.chdir(cwd)
#     return git_branch


# def get_git_hash():
#     """Get the current git hash.

#     Get the current git hash.

#     Parameters
#     ----------
#     None

#     Returns
#     -------
#     git_hash : str
#         Hash for current commit

#     Raises
#     ------
#     subprocess.CalledProcessError
#         If unable to determine git hash
#     """
#     cwd = os.getcwd()
#     os.chdir(os.environ['PINN_ROOT'])
#     cmd = 'git rev-parse HEAD'
#     cproc = subprocess.run(cmd, shell=True, check=True, text=True,
#                            capture_output=True)
#     git_hash = str(cproc.stdout.rstrip())  # Originally bytes
#     os.chdir(cwd)
#     return git_hash


# ----------------------------------------------------------------------------

# Tensorflow utilities


# def disable_gpus():
#     """Tell TensorFlow not to use GPU.

#     Tell TensorFlow not to use GPU.

#     Parameters
#     ----------
#     None

#     Returns
#     -------
#     None

#     Raises
#     ------
#     AssertionError : If this code cannot disable a GPU.
#     """
#     # Disable all GPUS.
#     tf.config.set_visible_devices([], 'GPU')

#     # Make sure the GPU were disabled.
#     visible_devices = tf.config.get_visible_devices()
#     for device in visible_devices:
#         assert device.device_type != 'GPU'


# ----------------------------------------------------------------------------

# Neural network utilities

# Initial parameter ranges
# W0_RANGE = [-0.1, 0.1]  # Hidden layer weights
# U0_RANGE = [-0.1, 0.1]  # Hidden layer biases
# V0_RANGE = [-0.1, 0.1]  # Output layer weights


# def build_model(n_layers, n_hidden, activation):
#     """Build a multi-layer neural network model.

#     Build a fully-connected, multi-layer neural network with single output.
#     Each layer will have H hidden nodes. Each hidden node has weights and
#     a bias, and uses the specified activation function. The output layer
#     does not use a bias.

#     Weights and biases are initialized with a uniformed rndom distribution
#     in the ranges defined in W0_RANGE, U0_RANGE, and V0_RANGE.

#     The number of inputs is determined when the network is first used.

#     Parameters
#     ----------
#     n_layers : int
#         Number of hidden layers to create.
#     n_hidden : int
#         Number of nodes to use in each hidden layer.
#     activation : str
#         Name of activation function (from TensorFlow) to use.

#     Returns
#     -------
#     model : tf.keras.Sequential
#         The neural network.

#     Raises
#     ------
#     None
#     """
#     layers = []
#     for _ in range(n_layers):
#         hidden_layer = tf.keras.layers.Dense(
#             units=n_hidden, use_bias=True,
#             activation=tf.keras.activations.deserialize(activation),
#             kernel_initializer=tf.keras.initializers.RandomUniform(*W0_RANGE),
#             bias_initializer=tf.keras.initializers.RandomUniform(*U0_RANGE)
#         )
#         layers.append(hidden_layer)
#     output_layer = tf.keras.layers.Dense(
#         units=1,
#         activation=tf.keras.activations.linear,
#         kernel_initializer=tf.keras.initializers.RandomUniform(*V0_RANGE),
#         use_bias=False,
#     )
#     layers.append(output_layer)
#     model = tf.keras.Sequential(layers)
#     return model


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


# def import_problem(problem_path):
#     """Import the Python file which defines the problem to solve.

#     Import the Python file which defines the problem to solve.

#     Note that the absolute path is required by module_from_spec().

#     Parameters
#     ----------
#     problem_path : str
#         Path to problem definition file.

#     Returns
#     -------
#     p : module
#         Module object for problem definition.

#     Raises
#     ------
#     None
#     """
#     abspath = os.path.abspath(problem_path)
#     problem_name = os.path.splitext(os.path.split(abspath)[-1])[-2]
#     spec = importlib.util.spec_from_file_location(problem_name, abspath)
#     p = importlib.util.module_from_spec(spec)
#     sys.modules[problem_name] = p
#     spec.loader.exec_module(p)
#     return p


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


# def read_grid_file(path):
#     """Read grid description and data from a file.

#     Read the grid description and data from a file. The file is assumed to
#     contain a grid header, followed list of nrows points. Each point
#     definines a grid location (x0, x1, ...) and zero or more values defined
#     at that location (y0, y1, ...).

#     Parameters
#     ----------
#     path : str
#         Path to grid file

#     Returns
#     -------
#     column_descriptions : list of ncols tuples, each (str, float, float, int)
#         List of (varname, min, max, n) for each grid dimension.
#     data : np.ndarray, shape (nrows, ncols)
#         Numpy array of data in file

#     Raises
#     ------
#     AssertionError
#         If this is not a grid file
#     """
#     # Read the grid description.
#     column_descriptions = []
#     COMMENT_PREFIX = '# '
#     COMMENT_PREFIX_LEN = len(COMMENT_PREFIX)
#     GRID_HEADER_LINE = f"{COMMENT_PREFIX}GRID"
#     with open(path, 'r', encoding='utf-8') as f:

#         # Make sure this is a grid file.
#         line = f.readline().rstrip()
#         assert line == GRID_HEADER_LINE

#         # Read the names of the independent variables (dimensions) which
#         # define the grid points.
#         line = f.readline().rstrip()
#         assert line.startswith(COMMENT_PREFIX)
#         dim_names_str = line[COMMENT_PREFIX_LEN:]
#         dim_names = dim_names_str.split(' ')
#         n_dim = len(dim_names)
#         assert n_dim > 0

#         # Read the grid description.
#         line = f.readline().rstrip()
#         assert line.startswith(COMMENT_PREFIX)
#         description_str = line[COMMENT_PREFIX_LEN:]
#         fields = description_str.split(' ')
#         xmin = [float(f) for f in fields[::3]]
#         xmax = [float(f) for f in fields[1::3]]
#         nx = [int(f) for f in fields[2::3]]

#         # Read the column names.
#         line = f.readline().rstrip()
#         assert line.startswith(COMMENT_PREFIX)
#         column_names_str = line[COMMENT_PREFIX_LEN:]
#         column_names = column_names_str.split(' ')
#         n_cols = len(column_names)
#         n_var = n_cols - n_dim
#         pad = [None]*n_var
#         column_descriptions = [
#             (s1, f1, f2, i1)
#             for (s1, f1, f2, i1) in
#             zip(column_names, xmin + pad, xmax + pad, nx + pad)
#         ]

#     # Now load the data table.
#     data = np.loadtxt(path)

#     # Return the grid description and data.
#     return column_descriptions, data
