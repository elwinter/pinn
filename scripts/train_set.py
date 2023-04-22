#!/usr/bin/env python


"""Train a set of PINN models.

Train a set of PINN models.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.
import argparse
import os

# Import 3rd-party modules.

# Import project modules.


# Program constants

# Program description string for help text.
DESCRIPTION = "Train a set of PINN models."

# Use clock to seed random number generator.
DEFAULT_SEEDS = "CLOCK"

# # Define the range of random number seeds to use.
# seeds = list(range(5, 6))
# print(f"seeds = {seeds}")

# # Define the PINN code root.
# PINN_ROOT = os.path.join(
#     os.environ["HOME"], "research_local", "src", "pinn"
# )
# print(f"PINN_ROOT = {PINN_ROOT}")

# # Specify the branch path.
# BRANCH = "periodic_save_model"
# BRANCH_PATH = os.path.join(PINN_ROOT, BRANCH, "pinn")
# print(f"BRANCH_PATH = {BRANCH_PATH}")

# # Define problem location.
# PROBLEM_CLASS = "loop2d"
# PROBLEM_NAME = "loop2d_BxBy"
# PROBLEM_ROOT = os.path.join(
#     BRANCH_PATH, "problems", PROBLEM_CLASS, PROBLEM_NAME
# )
# print(f"PROBLEM_ROOT = {PROBLEM_ROOT}")

# # Define PINN command.
# PINN_CMD = os.path.join(BRANCH_PATH, "pinn", "pinn1.py")
# print(f"PINN_CMD = {PINN_CMD}")

# # Specify the command template.
# CMD_TEMPLATE = (
#     "{{ pinn_cmd }}"
#     " --debug"
#     " --verbose"
#     " --seed={{ seed }}"
#     " --max_epochs={{ max_epochs }}"
#     " --save_model={{ save_model }}"
#     " --n_layers={{ n_layers }}"
#     " --n_hid={{ n_hid }}"
#     " --data={{ data_path }}"
#     " -w={{ w }}"
#     " {{ problem_path }}"
#     " {{ training_points_path }}"
# )
# print(f"CMD_TEMPLATE = {CMD_TEMPLATE}")
# cmd_template = Template(CMD_TEMPLATE)

# # Specify problem files.
# data_path = os.path.join(
#     PROBLEM_ROOT, f"{PROBLEM_NAME}_initial_conditions.dat"
# )
# problem_path = os.path.join(
#     PROBLEM_ROOT, f"{PROBLEM_NAME}.py"
# )
# training_points_path = os.path.join(
#     PROBLEM_ROOT, f"{PROBLEM_NAME}_training_grid.dat"
# )

# # Specify standard options for the set.
# options = {
#     "pinn_cmd": PINN_CMD,
#     "max_epochs": 100,
#     "save_model": 50,
#     "n_layers": 4,
#     "n_hid": 100,
#     "data": data_path,
#     "w": 0.95,
#     "problem_path": problem_path,
#     "training_points_path": training_points_path,
# }

# original_cwd = os.getcwd()

# for s in seeds:
#     print("==========")
#     print(f"Performing run for seed = {s}")
#     run_path = str(s)
#     os.mkdir(run_path)
#     options["seed"] = s
#     cmd = cmd_template.render(options)
#     print(f"cmd = {cmd}")
#     os.chdir(run_path)
#     with open("cmd", "w") as f:
#         f.write(cmd)
#     os.system(cmd)
#     os.chdir(original_cwd)


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
        "-d", "--debug", action="store_true", default=False,
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--seeds", type=str, default=DEFAULT_SEEDS,
        help="Random number generator seeds (comma-separated integers), or "
             "'CLOCK' for time-based seeds (default: %(default)s)"
    )
    parser.add_argument(
        "--set_directory", type=str, default=os.getcwd(),
        help="Directory to contain trained models (default: %(default)s)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Print verbose output (default: %(default)s)."
    )
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
    debug = args.debug
    seeds_str = args.seeds
    set_directory = args.set_directory
    verbose = args.verbose
    if debug:
        print(f"args = {args}")
        print(f"debug = {debug}")
        print(f"seeds_str = {seeds_str}")
        print(f"set_directory = {set_directory}")
        print(f"verbose = {verbose}")

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


if __name__ == "__main__":
    """Begin main program."""
    main()
