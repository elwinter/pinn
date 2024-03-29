#!/usr/bin/env python

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
options["pinn_root"] = os.environ["RESEARCH_INSTALL_DIR"]
options["python_environment"] = "research-3.10"

# Number of runs per data weight value.
N_RUNS = 1

# Range of data weights to use.
W_MIN = 0.00
W_MAX = 1.00
N_W = 11
WS = np.linspace(W_MIN, W_MAX, N_W)

# PBS job control constants
RUN_JOB_COMMAND = "bash"
# RUN_JOB_COMMAND = "qsub"

# PBS job constants (for derecho)
options["pbs_account"] = "UJHB0019"
options["pbs_queue"] = "main"
options["pbs_walltime"] = "00:05:00"
options["pbs_select"] = "select=1:ncpus=128"

# Options for problem set
options["problem_name"] = "bw3d"

# Options for pinn1.py for all runs.
options["pinn1_activation"] = "sigmoid"
options["pinn1_batch_size"] = -1
# options["pinn1_clobber"] = False
# options["pinn1_convcheck"] = False
# options["pinn1_debug"] = False
options["pinn1_learning_rate"] = 0.01
options["pinn1_max_epochs"] = 10000
options["pinn1_n_hid"] = 10
options["pinn1_n_layers"] = 1
options["pinn1_precision"] = "float32"
options["pinn1_save_model"] = 1000
# options["pinn1_save_weights"] = False
options["pinn1_seed"] = None
# options["tolerance"] = 1e-3
# options["verbose"] = True
# options["validation"] = "/homes/winteel1/mollie/research/src/pinn/development/pinn/problems/bw3d/bw3d/bw3d_validation.dat"
# options["verbose"] = True
options["pinn1_w_data"] = None
options["pinn1_problem"] = "/homes/winteel1/mollie/research/src/pinn/development/pinn/problems/bw3d/bw3d/bw3d.py"
options["pinn1_training_points"] = "/homes/winteel1/mollie/research/src/pinn/development/pinn/problems/bw3d/bw3d/bw3d_011_021_021_021_training_grid.dat"
options["pinn1_data"] = "/homes/winteel1/mollie/research/src/pinn/development/pinn/problems/bw3d/bw3d/bw3d_011_021_021_021_data.dat"
    
# Read and create the PBS script template.
PBS_TEMPLATE_FILE = "/homes/winteel1/mollie/research/src/pinn/development/pinn/problems/bw3d/bw3d/bw3d-template.pbs"
with open(PBS_TEMPLATE_FILE) as f:
    pbs_template_content = f.read()
pbs_template = Template(pbs_template_content)


def main():
    """Begin main program."""

    # Save the starting directory.
    start_dir = os.getcwd()
    print(f"start_dir = {start_dir}")

    # Perform all runs.
    for w in WS:
        print(f"w = {w:.2f}")

        # Make a directory for runs using this weight, and go there.
        w_dirname = f"w={w:.2f}"
        print(f"w_dirname = {w_dirname}")
        os.mkdir(w_dirname)
        os.chdir(w_dirname)
        weight_dir = os.getcwd()
        print(f"Current directory is {weight_dir}.")

        # Perform runs for this data weight.
        for run in range(N_RUNS):
            # print(f"run = {run}")

            # Compute the seed as an integer number of microseconds.
            seed = datetime.datetime.now().timestamp()
            seed = int(seed*MICROSECONDS_PER_SECOND)
            # print(f"seed = {seed}")

            # Assemble the run ID string.
            run_id = f"{options['problem_name']}-{run:02d}-{seed}"
            print(f"run_id = {run_id}")

            # Make a directory for this run_id, and go there.
            os.mkdir(run_id)
            os.chdir(run_id)
            cwd = os.getcwd()
            print(f"Current directory is {cwd}.")

            # Update the options for this run.
            options["run_id"] = run_id
            options["pinn1_seed"] = seed
            options["pinn1_w_data"] = w

            # Render the template for this run and save as a file.
            pbs_content = pbs_template.render(options)
            # print(f"pbs_content = {pbs_content}")
            pbs_file = f"{run_id}.pbs"
            with open(pbs_file, "w") as f:
                f.write(pbs_content)

            # Submit the job and save the output.
            args = [RUN_JOB_COMMAND, pbs_file]
            subprocess.run(args, check=True)
            # result = subprocess.run(
            #     args, check=True,
            #     stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            # )
            # with open("job.out", "w") as f:
            #     f.write(result.stdout.decode())

            # Move back to the weight directory.
            os.chdir(weight_dir)
            cwd = os.getcwd()
            print(f"Current directory is {cwd}.")

        # Move back to the top directory.
        os.chdir(start_dir)
        cwd = os.getcwd()
        print(f"Current directory is {cwd}.")


if __name__ == "__main__":
    main()
