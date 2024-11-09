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

# Import project modules.


# ----------------------------------------------------------------------------

# Command-line utilities


# Default values for command-line arguments.
DEFAULT_ARGUMENTS = {
    "activation": "sigmoid",
    "batch_size": -1,
    "clobber": False,
    "debug": False,
    "learning_rate": 0.01,
    "load_model": None,
    "max_epochs": 100,
    "multi": False,
    "n_hid": 10,
    "n_layers": 1,
    "nogpu": False,
    "precision": "float32",
    "save_model": -1,
    "seed": 0,
    "verbose": False,
}


def create_minimal_command_line_parser(description):
    """Create a minimal command-line parser.

    Create a minimal command-line parser. It just adds the --debug and
    --verbose options.

    Parameters
    ----------
    description : str
        Script description.

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
        default=DEFAULT_ARGUMENTS["debug"],
        help="Print debugging output (default: %(default)s)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        default=DEFAULT_ARGUMENTS["verbose"],
        help="Print verbose output (default: %(default)s)."
    )
    return parser


def create_neural_network_command_line_argument_parser(description):
    """Create a command-line argument parser for neural network code.

    Create a command-line argument parser for neural network code. This
    code builds on the minimal command-line argument parser by adding a set
    of options for neural network code in the pinn package.

    Parameters
    ----------
    description : str
        Script description.

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
        "--activation", "-a",
        default=DEFAULT_ARGUMENTS["activation"],
        help="Activation function for hidden nodes (default: %(default)s)"
    )
    parser.add_argument(
        "--batch_size", type=int,
        default=DEFAULT_ARGUMENTS["batch_size"],
        help="Batch size (-1 for single batch) (default: %(default)s)"
    )
    parser.add_argument(
        "--clobber", action="store_true",
        help="Overwrite existing output directory (default: %(default)s)"
    )
    parser.add_argument(
        "--learning_rate", type=float,
        default=DEFAULT_ARGUMENTS["learning_rate"],
        help="Initial learning rate for training (default: %(default)s)"
    )
    parser.add_argument(
        "--load_model",
        default=DEFAULT_ARGUMENTS["load_model"],
        help="Path to directory containing models to load (default:"
             " %(default)s)"
    )
    parser.add_argument(
        "--max_epochs", type=int,
        default=DEFAULT_ARGUMENTS["max_epochs"],
        help="Maximum number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--multi", action="store_true",
        help="Use a single multi-output network (default: %(default)s)"
    )
    parser.add_argument(
        "--n_hid", type=int,
        default=DEFAULT_ARGUMENTS["n_hid"],
        help="Number of hidden nodes per layer (default: %(default)s)"
    )
    parser.add_argument(
        "--n_layers", type=int,
        default=DEFAULT_ARGUMENTS["n_layers"],
        help="Number of hidden layers (default: %(default)s)"
    )
    parser.add_argument(
        "--nogpu", action="store_true",
        help="Disable TensorFlow use of GPU(s) (default: %(default)s)"
    )
    parser.add_argument(
        "--precision", type=str,
        default=DEFAULT_ARGUMENTS["precision"],
        help="Precision to use in TensorFlow solution (default: %(default)s)"
    )
    parser.add_argument(
        "--save_model", type=int,
        default=DEFAULT_ARGUMENTS["save_model"],
        help="Save interval (epochs) for trained model (0 = do not save, "
        "-1 = save at end, n > 0 = save every n epochs) (default: %(default)s)"
    )
    parser.add_argument(
        "--seed", type=int,
        default=DEFAULT_ARGUMENTS["seed"],
        help="Seed for random number generator (default: %(default)s)"
    )

    # Return the parser.
    return parser


# ----------------------------------------------------------------------------

# General program utilities


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


def save_arguments(args: dict, output_dir: str):
    """Save the program arguments.

    Save a record of the program arguments as the file "arguments.py" in the
    specified directory. The file is an importable python module defining the
    dict "args".

    Parameters
    ----------
    args : dict
        dict of command-line arguments.
    output_dir : str
        Path to directory to contain the report.

    Returns
    -------
    path : str
        Path to arguments file.

    Raises
    ------
    None
    """
    path = os.path.join(output_dir, "arguments.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"args = {repr(args)}")
    return path


def save_system_information(output_dir):
    """Save a summary of system characteristics.

    Save a summary of the host system hardware and software in the specified
    directory.

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
    path = os.path.join(output_dir, "system_information.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("System information:\n")
        f.write(f"Start time: {datetime.datetime.now()}\n")
        f.write(f"Host name: {platform.node()}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"uname: {' '.join(platform.uname())}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Python build: {' '.join(platform.python_build())}\n")
        f.write(f"Python compiler: {platform.python_compiler()}\n")
        f.write(f"Python implementation: {platform.python_implementation()}\n")
        f.write(f"conda environment: {os.environ['CONDA_DEFAULT_ENV']}\n")
        f.write(f"NumPy version: {np.__version__}\n")
        f.write(f"TensorFlow version: {tf.__version__}\n")
        f.write("Available TensorFlow devices: \n")
        for (i_device, device) in enumerate(device_lib.list_local_devices()):
            f.write(f"Device {i_device}: device {device}")
        f.write(f"Git branch: {get_git_branch()}\n")
        f.write(f"Git hash: {get_git_hash()}\n")
    return path


def load_problem_data(data_path: str, precision: str):
    """Load the problem data.

    Load the problem data, which specifies the function values at each
    point.

    Each of the n_data lines contains a tuple of n_dim coordinates (the
    independent variable values for a location), followed by a tuple of n_var
    dependent variables, containing the values of each dependent variable at
    that location.

    If the data file contains a single point or column, the array is reshaped
    to 2-D (shape (n_data, 1) to be compatible with TensorFlow.

    Parameters
    ----------
    data_path : str
        Path to Numpy-format text data file.
    precision : str
        Precision to use when creating NumPy array for data. The valid values
        are defined as TensorFlow data types, which usually are the same as
        NumPy data types.

    Returns
    -------
    XY : np.ndarray, shape(n_data, n_dim + n_var)
        Array of data points.

    Raises
    ------
    None
    """
    # Load the data.
    XY = np.loadtxt(data_path, dtype=precision)

    # If the problem data shape is 1-D (a single point), reshape to 2-D,
    # (n_data, 1) to make compatible with later TensorFlow calls, which
    # expect a 2D Tensor.
    if len(XY.shape) == 1:
        XY = XY.reshape(XY.shape[0], 1)

    # Return the problem data.
    return XY


def read_grid_file(path: str):
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
    COMMENT_PREFIX = "# "
    COMMENT_PREFIX_LEN = len(COMMENT_PREFIX)
    GRID_HEADER_LINE = f"{COMMENT_PREFIX}GRID"
    with open(path, "r", encoding="utf-8") as f:

        # Make sure this is a grid file.
        line = f.readline().rstrip()
        assert line == GRID_HEADER_LINE

        # Read the column descriptions.
        line = f.readline().rstrip()
        assert line.startswith(COMMENT_PREFIX)
        line = line.lstrip("# ")
        fields = line.split()
        vnames = fields[::4]
        nvar = len(vnames)
        vmin = [float(x) for x in fields[1::4]]
        vmax = [float(x) for x in fields[2::4]]
        nv = [int(n) for n in fields[3::4]]
        column_descriptions = {}
        column_descriptions["name"] = vnames
        column_descriptions["min"] = vmin
        column_descriptions["max"] = vmax
        column_descriptions["n"] = nv

    # Now load the data table.
    data = np.loadtxt(path)

    # Return the grid description and data.
    return column_descriptions, data


# ----------------------------------------------------------------------------

# git utilities


def get_git_branch():
    """Get the current git branch.

    Get the current git branch of the pinn code using the PINN_ROOT
    environment variable.

    Parameters
    ----------
    None

    Returns
    -------
    git_branch : str
        Name of current git branch for active pinn code.

    Raises
    ------
    subprocess.CalledProcessError
        If unable to determine git branch.
    """
    cwd = os.getcwd()
    this_dir = os.path.split(os.path.realpath(__file__))[0]
    os.chdir(this_dir)
    cmd = "git branch"
    cproc = subprocess.run(cmd, shell=True, check=True, text=True,
                           capture_output=True)
    lines = cproc.stdout.splitlines()
    git_branch = None
    for line in lines:
        if line.startswith("*"):
            git_branch = line[2:]
    if git_branch is None:
        raise subprocess.CalledProcessError(
            "Unable to determine git branch!", cmd)
    os.chdir(cwd)
    return git_branch


def get_git_hash():
    """Get the current git hash.

    Get the current git hash of the pinn code using the PINN_ROOT environment
    variable.

    Parameters
    ----------
    None

    Returns
    -------
    git_hash : str
        Hash for current commit of active pinn code.

    Raises
    ------
    None
    """
    cwd = os.getcwd()
    this_dir = os.path.split(os.path.realpath(__file__))[0]
    os.chdir(this_dir)
    cmd = "git rev-parse HEAD"
    cproc = subprocess.run(cmd, shell=True, check=True, text=True,
                           capture_output=True)
    git_hash = str(cproc.stdout.rstrip())  # Originally bytes
    os.chdir(cwd)
    return git_hash


# ----------------------------------------------------------------------------

# Tensorflow utilities


def configure_tensorflow(nogpu: bool, precision: str, seed: int):
    """Configure TensorFlow.

    Configure TensorFlow. Configuration optionally deactivates GPU use, and
    sets float precision and the random number seed.

    Parameters
    ----------
    nogpu : bool
        True to not use a GPU for TensorFlow calculations.
    precision : str
        Precision to use for TensorFlow calculations.
    seed : int
        Seed for TensorFlow random number generator.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # If requested, disable TensorFlow use of GPU.
    if nogpu:
        disable_gpus()

    # Set the backend TensorFlow precision.
    tf.keras.backend.set_floatx(precision)

    # Set the random number seed for reproducibility.
    tf.random.set_seed(seed)


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


# ----------------------------------------------------------------------------

# Neural network utilities


def create_batches(X_train: np.ndarray, batch_size: int):
    """Split the data into batches of tf.Variable.

    Split the data into batches of tf.Variable.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, p.n_dim)
        Array of training points.
    batch_size : int
        Nominal number of points in 1 batch.

    Returns
    -------
    training_batches : list of tf.Variable
        Training points split into batches.

    Raises
    ------
    None
    """
    training_batches = []
    n_train = X_train.shape[0]
    if batch_size == -1:
        Xb_tf = tf.Variable(X_train)
        training_batches.append(Xb_tf)
    else:
        n_batches = int(np.ceil(n_train/batch_size))
        for ib in range(n_batches):
            i_start = ib*batch_size
            i_end = (ib + 1)*batch_size
            i_end = min(i_end, n_train)
            Xb_np = X_train[i_start:i_end, ...]
            Xb_tf = tf.Variable(Xb_np)
            training_batches.append(Xb_tf)

    # Return the list of batches.
    return training_batches


# Initial network parameter ranges.
W0_RANGE = [-0.1, 0.1]  # Hidden layer weights
U0_RANGE = [-0.1, 0.1]  # Hidden layer biases
V0_RANGE = [-0.1, 0.1]  # Output layer weights


def create_model(n_layers: int, n_hid: int, activation: str):
    """Create a multi-layer neural network model.

    Create a fully-connected, n_layers-hidden-layer neural network with a
    single output. Each layer will have n_hid hidden nodes. Each hidden node
    has weights and a bias, and uses the specified activation function. The
    output layer uses a linear transfer function and no bias.

    Weights and biases are initialized with a uniform random distribution
    in the ranges defined in W0_RANGE, U0_RANGE, and V0_RANGE.

    The number of inputs is determined when the network is first used.

    Parameters
    ----------
    n_layers : int
        Number of hidden layers to create.
    n_hid : int
        Number of nodes to use in each hidden layer.
    activation : str
        Name of activation function (from TensorFlow) to use in each hidden
        node.

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
            units=n_hid, use_bias=True,
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


def create_models(variable_names: list, n_layers: int, n_hid: int,
                  activation: str, multi: bool):
    """Create untrained models.

    Create untrained models.

    Parameters
    ----------
    variable_names : list of str
        List of variable names, one variable per model.
    n_layers : int
        Number of hidden layers to create.
    n_hid : int
        Number of nodes to use in each hidden layer.
    activation : str
        Name of activation function (from TensorFlow) to use.
    multi : bool
        True to create a single multi-output model.

    Returns
    -------
    models : list of keras.src.engine.sequential.Sequential
        The untrained models.

    Raises
    ------
    None
    """
    models = []
    if multi:
        raise TypeError("--multi not supported!")
        # model = create_multi_output_model(n_layers, n_hid, activation,
        #                                   len(variable_names))
        # models.append(model)
    else:
        for _ in variable_names:
            model = create_model(n_layers, n_hid, activation)
            models.append(model)

    # Return the models.
    return models


def create_multi_output_model(n_layers: int, n_hid: int, activation: str,
                              n_out: bool):
    """Create a multi-output, multi-layer neural network model.

    Create a fully-connected, n_layers-layer neural network with multiple
    n_out  outputs. Each layer will have n_hid hidden nodes. Each hidden node
    has weights and a bias, and uses the specified activation function.

    The number of inputs is determined when the network is first used.

    Parameters
    ----------
    n_layers : int
        Number of hidden layers to create.
    n_hid : int
        Number of nodes to use in each hidden layer.
    activation : str
        Name of activation function (from TensorFlow) to use.
    n_out : int
        Number of network outputs

    Returns
    -------
    model : tf.keras.Sequential
        The neural network.
    """
    layers = []
    for _ in range(n_layers):
        hidden_layer = tf.keras.layers.Dense(
            units=n_hid, use_bias=True,
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


def create_optimizer(learning_rate: float):
    """Create the training optimizer.

    Create the training optimizer.

    Parameters
    ----------
    learning_rate: float
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
    return optimizer


def find_last_epoch(results_path: str):
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


def load_models(model_directory: str, variable_names: list, multi: bool):
    """Load trained models.

    Load trained models.

    Parameters
    ----------
    model_directory : str
        Path to directory containing trained models.
    variable_names : list of str
        List of variable names, one variable per model.
    multi : bool
        True if a multi-output model is desired.

    Returns
    -------
    models : list of keras.src.engine.sequential.Sequential
        The trained models.

    Raises
    ------
    None
    """
    models = []
    if multi:
        raise TypeError("--multi not supported!")
        # path = os.path.join(model_directory, "model")
        # model = tf.keras.models.load_model(path)
        # models.append(model)
    else:
        for vname in variable_names:
            path = os.path.join(model_directory, f"model_{vname}")
            model = tf.keras.models.load_model(path)
            models.append(model)

    # Return the models.
    return models


def save_models(models: list, model_directory: str, epoch: int,
                variable_names: list, multi: bool):
    """Save trained models.

    Save trained models.

    Parameters
    ----------
    models : list of keras.src.engine.sequential.Sequential
        Models to save.
    model_directory : str
        Path to directory containing trained models.
    epoch : int
        Epochs used to train model.
    variable_names : list of str
        List of variable names, one variable per model.
    multi : bool
        True if a multi-output model is desired.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    if multi:
        raise TypeError("--multi not supported!")
        # path = os.path.join(
        #     output_dir, "models", f"{epoch:06d}", "model_multi"
        # )
        # model = models[0]
        # model.save(path)
        # path = os.path.join(output_dir, "models", "model_multi.txt")
        # old_stdout = sys.stdout
        # with open(path, "w", encoding="utf-8") as f:
        #     sys.stdout = f
        #     model.summary()
        # sys.stdout = old_stdout
    else:
        for (i, model) in enumerate(models):
            variable_name = variable_names[i]
            save_dir = os.path.join(model_directory, "models", f"{epoch:06d}")
            path = os.path.join(save_dir, f"model_{variable_name}")
            model.save(path)
            path = os.path.join(save_dir, f"model_{variable_name}.txt")
            old_stdout = sys.stdout
            with open(path, "w", encoding="utf-8") as f:
                sys.stdout = f
                model.summary()
            sys.stdout = old_stdout


if __name__ == "__main__":
    pass
