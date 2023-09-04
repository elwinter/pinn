#!/usr/bin/env python

"""Train a set of pinn0 models for the lagaris01 problem.

Train a set of pinn0 models for the lagaris01 problem.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules
import argparse
import datetime
import os
import subprocess

# Import supplemental modules
from jinja2 import Template
import numpy as np

# Import project modules


# Program constants

# Program description
DESCRIPTION = "Train a set of pinn0 models for the lagaris01 problem."

# Minimum, maximum, and count of data weights.
W_MIN = 0.0
W_MAX = 1.0
N_W = 3

# Number of randomized runs for each set of hyperparameters.
N_RUNS = 2

# Path to script template.
SCRIPT_TEMPLATE = os.path.join(
    os.environ["PINN_INSTALL_DIR"],
    "problems", "lagaris", "lagaris01", "lagaris01_pinn0_template.pbs"
)

# Initialize the options dictionary used to populate the run script template.
options = {}
options["problem_name"] = "lagaris01"
# PBS job constants (for derecho)
options["pbs_account"] = "UJHB0019"
options["pbs_queue"] = "main"
options["pbs_walltime"] = "00:05:00"
options["pbs_select"] = "select=1:ncpus=128"
# Specify the software installation to use.
options["pinn_root"] = os.environ["PINN_INSTALL_DIR"]
options["python_environment"] = "research-3.10"
# Arguments for pinn code.
options["activation"] = "sigmoid"
options["clobber"] = ""
options["debug"] = ""
options["learning_rate"] = 0.01
options["max_epochs"] = 100
options["n_hid"] = 10
options["n_layers"] = 1
options["nogpu"] = ""
options["precision"] = "float32"
options["save_model"] = -1
options["seed"] = None
options["validation"] = ""
options["verbose"] = "--verbose"
options["problem_path"] = os.path.join(
    os.environ["PINN_INSTALL_DIR"],
    "problems", "lagaris", "lagaris01", "lagaris01.py"
)
options["data"] = os.path.join(
    os.environ["PINN_INSTALL_DIR"],
    "problems", "lagaris", "lagaris01", "data", "lagaris01_data.dat"
)


# General constants
MICROSECONDS_PER_SECOND = 1e6

# Command to run the job.
RUN_JOB_COMMAND = "bash"


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
    parser = argparse.ArgumentParser(DESCRIPTION)
    parser.add_argument(
        "-d", "--debug", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    return parser


def main():
    """Begin main program."""
    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    debug = args.debug
    verbose = args.verbose
    if debug:
        print(f"args = {args}", flush=True)

    # Create the range of weights.
    ws = np.linspace(W_MIN, W_MAX, N_W)
    if debug:
        print(f"ws = {ws}")

    # Read the script template.
    with open(SCRIPT_TEMPLATE) as f:
        script_template_content = f.read()
    script_template = Template(script_template_content)

    # Save the starting directory.
    start_dir = os.getcwd()
    if debug:
        print(f"start_dir = {start_dir}")

    # ------------------------------------------------------------------------

    # Perform all runs.
    for w in ws:
        if debug:
            print(f"w = {w:.2f}")

        # Make a directory for runs using this weight, and go there.
        w_dirname = f"w={w:.2f}"
        if debug:
            print(f"w_dirname = {w_dirname}")
        os.mkdir(w_dirname)
        os.chdir(w_dirname)
        weight_dir = os.getcwd()
        if debug:
            print(f"weight_dir = {weight_dir}")

        # Perform a set of duplicate runs for this data weight.
        # Use the current time to set the random number generator random_seed.
        for run in range(N_RUNS):
            if debug:
                print(f"run = {run}")

            # Compute the random_seed as an integer number of microseconds for the
            # current time.
            seed = datetime.datetime.now().timestamp()
            seed = int(seed*MICROSECONDS_PER_SECOND)
            if debug:
                print(f"seed = {seed}")

            # Assemble the run ID string.
            run_id = f"{options['problem_name']}-{run:02d}-{seed}"
            if debug:
                print(f"run_id = {run_id}")

            # Make a directory for this run_id, and go there.
            # THIS SHOULD NEVER FAIL SINCE TIME MOVES FORWARD ONLY!
            os.mkdir(run_id)
            os.chdir(run_id)
            cwd = os.getcwd()
            if debug:
                print(f"cwd = {cwd}")

            # Update the options for this run.
            options["run_id"] = run_id
            options["seed"] = seed

            # Render the template for this run and save as a file.
            script_content = script_template.render(options)
            if debug:
                print(f"script_content = {script_content}")
            script_file = f"{run_id}.sh"
            with open(script_file, "w") as f:
                f.write(script_content)

            # Submit the job and save the output.
            args = [RUN_JOB_COMMAND, script_file]
            result = subprocess.run(
                args, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            with open("job.out", "w") as f:
                f.write(result.stdout.decode())

            # Move back to the weight directory.
            os.chdir(weight_dir)
            cwd = os.getcwd()
            if debug:
                print(f"cwd = {cwd}")

        # Move back to the top directory.
        os.chdir(start_dir)
        cwd = os.getcwd()
        if debug:
            print(f"cwd = {cwd}")


if __name__ == "__main__":
    main()
