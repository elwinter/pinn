#!/usr/bin/env python

"""Perform a set of pinn1.py runs.

Perform a set of pinn1.py runs.

A set is typically composed of a series of data weight values, each of which
gets a number of runs varying only in the random number seed.

This code requires that the PINN_INSTALL_DIR environment variable is set.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""
# Import standard modules
import datetime
import os
import subprocess

# Import supplemental modules
from jinja2 import Template
import numpy as np

# Import project modules


# General constants
MICROSECONDS_PER_SECOND = 1e6


# Initialize the options dictionary.
options = {}

# Specify the software installation to use.
options['pinn_root'] = os.environ['PINN_ROOT']

# Options for problem to solve
options['problem_class'] = 'lagaris'
options['problem_name'] = 'lagaris01'
options['problem_root'] = os.path.join(
    options['pinn_root'], 'problems',
    options['problem_class'], options['problem_name']
)
options['pinn1_problem_path'] = os.path.join(
    options['problem_root'], f"{options['problem_name']}.py"
)
options['pinn1_data_path'] = os.path.join(
    options['problem_root'], 'data', f"{options['problem_name']}_data.dat"
)
options['pinn1_training_path'] = os.path.join(
    options['problem_root'], 'data', f"{options['problem_name']}_training_grid.dat"
)
options['pinn1_results_dir'] = f"{options['problem_name']}-pinn1"
options['pinn1_plot_cmd'] = os.path.join(
    options['problem_root'], f"{options['problem_name']}_pinn1_plots.py"
)

# Specify the python environment for the runs.
options['python_environment'] = 'research-3.10'


# Range of data weights to use.
W_MIN = 0.2
W_MAX = 0.2
N_W = 1
WS = np.linspace(W_MIN, W_MAX, N_W)

# Number of runs per data weight value.
N_RUNS = 1

# Commands to run individual jobs
RUN_JOB_COMMAND = 'bash'

# PBS job constants (for derecho)
# RUN_JOB_COMMAND = "qsub"
# options["pbs_account"] = "UJHB0019"
# options["pbs_queue"] = "main"
# options["pbs_walltime"] = "00:05:00"
# options["pbs_select"] = "select=1:ncpus=128"

# Options for all runs.
# Use flag strings for binary options, or empty string ''.
# Set value to None for run-specific options that must be filled in.
options['pinn1_activation'] = 'sigmoid'
options['pinn1_debug'] = ''
options['pinn1_learning_rate'] = 0.01
options['pinn1_load_model'] = ''
options['pinn1_max_epochs'] = 1000
options['pinn1_n_hid'] = 10
options['pinn1_n_layers'] = 1
options['pinn1_nogpu'] = ''
options['pinn1_precision'] = 'float32'
options['pinn1_save_model'] = -1
options['pinn1_seed'] = None
options['pinn1_verbose'] = '--verbose'
options['pinn1_w_data'] = None

# Read and create the PBS script template.
PBS_TEMPLATE_FILE = os.path.join(
    options['pinn_root'], 'templates', 'pinn1_template.pbs'
)
with open(PBS_TEMPLATE_FILE) as f:
    pbs_template_content = f.read()
pbs_template = Template(pbs_template_content)


def main():
    """Begin main program."""

    # Print problem information.
    t_start = datetime.datetime.now()
    print(f"Run started at {t_start}")
    print(f"Problem class: {options['problem_class']}")
    print(f"Problem name: {options['problem_name']}")
    print(f"Problem root: {options['problem_root']}")
    print(f"Problem definition file: {options['pinn1_problem_path']}")
    print(f"Problem data file: {options['pinn1_data_path']}")
    print(f"Problem training points file: {options['pinn1_training_path']}")
    print(f"Plotting script for results: {options['pinn1_plot_cmd']}")
    print(f"PINN root: {options['pinn_root']}")
    print(f"Python environment: {options['python_environment']}")
    print(f"There will be {N_RUNS} runs for each weight in the set {WS}.")
    print(f"Each job will be submitted with {RUN_JOB_COMMAND}.")
    print(f"Script template: {PBS_TEMPLATE_FILE}")

    # Save the starting directory.
    start_dir = os.getcwd()
    print(f"start_dir = {start_dir}")

    # Perform all runs.
    for w in WS:
        w_str = f"{w:.2f}"
        print(f"Beginning runs for w = {w_str}.")

        # Make a directory for runs using this weight, and go there.
        w_dirname = f"w={w:.2f}"
        os.mkdir(w_dirname)
        os.chdir(w_dirname)
        weight_dir = os.getcwd()
        print(f"Current directory is {weight_dir}.")

        # Perform runs for this data weight.
        for run in range(N_RUNS):
            run_str = f"{run:02d}"
            print(f"Beginning run {run_str} for w = {w_str}.")

            # Compute the seed as an integer number of microseconds.
            seed = datetime.datetime.now().timestamp()
            seed = int(seed*MICROSECONDS_PER_SECOND)
            print(f"seed = {seed}")

            # Assemble the run ID string.
            options['run_id'] = f"{options['problem_name']}-{run:02d}-{seed}"
            print(f"run_id = {options['run_id']}")

            # Make a directory for this run_id, and go there.
            os.mkdir(options['run_id'])
            os.chdir(options['run_id'])
            cwd = os.getcwd()
            print(f"Current directory is {cwd}.")

            # Update the options for this run.
            options['pinn1_seed'] = seed
            options['pinn1_w_data'] = w

            # Render the template for this run and save as a file.
            pbs_content = pbs_template.render(options)
            pbs_file = f"{options['run_id']}.pbs"
            with open(pbs_file, 'w') as f:
                f.write(pbs_content)

            # Submit the job and save the output.
            print(f"Submitting run {options['run_id']} in directory {cwd}.")
            args = [RUN_JOB_COMMAND, pbs_file]
            result = subprocess.run(
                args, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            with open('job.out', 'w') as f:
                f.write(result.stdout.decode())

            # Move back to the weight directory.
            os.chdir(weight_dir)
            cwd = os.getcwd()
            print(f"Current directory is {cwd}.")

        # Move back to the top directory.
        os.chdir(start_dir)
        cwd = os.getcwd()
        print(f"Current directory is {cwd}.")

    # Print end-of-run information.
    t_stop = datetime.datetime.now()
    print(f"Run ended at {t_stop}")
    t_elapsed = t_stop - t_start
    print(f"Total run time: {t_elapsed}")


if __name__ == "__main__":
    """Call main loop."""
    main()
