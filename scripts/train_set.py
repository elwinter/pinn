#!/usr/bin/env python


"""Train a set of PINN models.

Train a set of PINN models. A "set" in this case means the same problem,
trained multiple times, using the same network and training hyperparameters,
each time using a different random number seed.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.
import argparse
import os

# Import 3rd-party modules.
from jinja2 import Template

# Import project modules.


# Program constants

# Program description string for help text.
DESCRIPTION = "Train a set of PINN models."

# Default activation function to use in hidden nodes.
DEFAULT_ACTIVATION = "sigmoid"

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

# Seeds for random number generator.
DEFAULT_SEEDS = "0"

# Default absolute tolerance for consecutive loss function values to indicate
# convergence.
DEFAULT_TOLERANCE = 1e-6

# Default normalized weight to apply to the boundary condition loss function.
DEFAULT_W_DATA = 0.0


# Define the PINN code root.
PINN_ROOT = os.environ["RESEARCH_INSTALL_DIR"]

# Define the PINN command.
PINN_CMD = os.path.join(PINN_ROOT, "pinn", "pinn1.py")

# Define the jinja2 command template.
CMD_TEMPLATE = (
    "{{ pinn_cmd }}"
    " --activation={{ activation }}"
    " {{ convcheck }}"
    " --data={{ data_path }}"
    " {{ debug }}"
    " --learning_rate={{ learning_rate }}"
    " --max_epochs={{ max_epochs }}"
    " --n_hid={{ n_hid }}"
    " --n_layers={{ n_layers }}"
    " --precision={{ precision }}"
    " --save_model={{ save_model }}"
    " {{ save_weights }}"
    " --seed={{ seed }}"
    " --tolerance={{ tolerance }}"
    " {{ verbose }}"
    " --w_data={{ w_data }}"
    " {{ problem_path }}"
    " {{ training_points_path }}"
    " >> pinn1.out"
)


def create_command_line_parser():
    """Create the command-line argument parser.

    Create the parser for command-line arguments.

    Parameters
    ----------
    None

    Returns
    -------
    parser : argparse.ArgumentParser
        Command-line argument parser for this script.

    Raises
    ------
    None
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--activation", "-a", default=DEFAULT_ACTIVATION,
        help="Specify activation function (default: %(default)s)."
    )
    parser.add_argument(
        "--convcheck", action="store_true",
        help="Perform convergence check (default: %(default)s)."
    )
    parser.add_argument(
        "--data_path", default=None,
        help="Path to optional input data file (default: %(default)s)."
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", default=False,
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
        help="Learning rate for training (default: %(default)s)"
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
        "--precision", type=str, default=DEFAULT_PRECISION,
        help="Precision to use in TensorFlow solution (default: %(default)s)"
    )
    parser.add_argument(
        "--save_model", type=int, default=DEFAULT_SAVE_MODEL,
        help="Save interval (epochs) for trained model (default: %(default)s)."
        " 0 = do not save, -1 = save at end, n > 0 = save every n epochs."
    )
    parser.add_argument(
        "--save_weights", action="store_true",
        help="Save the model weights at each epoch (default: %(default)s)."
    )
    parser.add_argument(
        "--seeds", type=str, default=DEFAULT_SEEDS,
        help="Random number generator seeds (comma-separated integers)"
             " (default: %(default)s)"
    )
    parser.add_argument(
        "--set_directory", type=str, default=os.getcwd(),
        help="Path to directory to contain trained models "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help="Absolute loss function convergence tolerance "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "--w_data", "-w", type=float, default=DEFAULT_W_DATA,
        help="Normalized weight for data loss function "
             "(default: %(default)s)."
    )
    parser.add_argument("problem_path", type=str)
    parser.add_argument("training_points_path", type=str)
    return parser


def main():
    """Begin main program.

    This is the top-level program code.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Set up the command-line parser.
    parser = create_command_line_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    activation = args.activation
    convcheck = args.convcheck
    data_path = args.data_path
    debug = args.debug
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    n_hid = args.n_hid
    n_layers = args.n_layers
    precision = args.precision
    save_model = args.save_model
    save_weights = args.save_weights
    seeds_str = args.seeds
    set_directory = args.set_directory
    tolerance = args.tolerance
    verbose = args.verbose
    w_data = args.w_data
    problem_path = args.problem_path
    training_points_path = args.training_points_path
    if debug:
        print(f"args = {args}")
        print(f"activation = {activation}")
        print(f"convcheck = {convcheck}")
        print(f"data_path = {data_path}")
        print(f"debug = {debug}")
        print(f"learning_rate = {learning_rate}")
        print(f"max_epochs = {max_epochs}")
        print(f"n_hid = {n_hid}")
        print(f"n_layers = {n_layers}")
        print(f"precision = {precision}")
        print(f"save_model = {save_model}")
        print(f"save_weights = {save_weights}")
        print(f"seeds_str = {seeds_str}")
        print(f"set_directory = {set_directory}")
        print(f"tolerance = {tolerance}")
        print(f"verbose = {verbose}")
        print(f"w_data = {w_data}")
        print(f"problem_path = {problem_path}")
        print(f"training_points_path = {training_points_path}")

    # If explicit random number generator seeds were specified, parse them.
    if seeds_str == "CLOCK":
        if verbose:
            print("Random number generator seeds will be generated from the "
                  "clock.")
    else:
        # NOTE: If seeds are provided, there must be at least as many seeds
        # as are needed to provide for all of the models in the set.
        if verbose:
            print("Parsing random number generator seeds.")
        seeds_str_list = seeds_str.split(",")
        seeds = [int(s) for s in seeds_str_list]
        if debug:
            print(f"seeds = {seeds}")

    # If the top-level directory for training the set is not found, create it.
    if os.path.isdir(set_directory):
        if verbose:
            print(f"Set directory {set_directory} exists.")
    else:
        if verbose:
            print(f"Set directory {set_directory} does not exist, creating.")
        os.makedirs(set_directory)

    # Create the command template.
    cmd_template = Template(CMD_TEMPLATE)
    if debug:
        print(f"cmd_template = {cmd_template}")

    # Assemble the options dictionary.
    options = {}
    options["pinn_cmd"] = PINN_CMD
    options["activation"] = activation
    if convcheck:
        options["convcheck"] = "--convcheck"
    options["data_path"] = data_path
    if debug:
        options["debug"] = "--debug"
    options["learning_rate"] = learning_rate
    options["max_epochs"] = max_epochs
    options["n_hid"] = n_hid
    options["n_layers"] = n_layers
    options["precision"] = precision
    options["save_model"] = save_model
    if save_weights:
        options["save_weights"] = "--save_weights"
    options["tolerance"] = tolerance
    if verbose:
        options["verbose"] = "--verbose"
    options["w_data"] = w_data
    options["problem_path"] = problem_path
    options["training_points_path"] = training_points_path

    # Train the model using each seed.
    for seed in seeds:
        print("==========")
        print(f"Performing training for seed = {seed}")

        # Create the directory to hold this run, then go there.
        run_path = os.path.join(set_directory, str(seed))
        if debug:
            print(f"run_path = {run_path}")
        os.makedirs(run_path)
        os.chdir(run_path)

        # Set the seed for this run.
        options["seed"] = seed

        # Render the template to create the command string.
        cmd = cmd_template.render(options)
        if debug:
            print(f"cmd = {cmd}")

        # Run the command.
        os.system(cmd)


if __name__ == "__main__":
    """Begin main program."""
    main()
