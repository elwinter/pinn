#!/usr/bin/env python

# Import standard modules
import datetime
import os
import subprocess

# Import supplemental modules
from jinja2 import Template
import numpy as np

# Import project modules


PROBLEM_CLASS = "waves1d"
PROBLEM_NAME = "eplasma3"
# print(f"PROBLEM_CLASS = {PROBLEM_CLASS}")
# print(f"PROBLEM_NAME = {PROBLEM_NAME}")

RESEARCH_ROOT = os.environ["RESEARCH_INSTALL_DIR"]
PROBLEMS_DIR = os.path.join(RESEARCH_ROOT, "problems")
PROBLEM_DIR = os.path.join(PROBLEMS_DIR, PROBLEM_CLASS, PROBLEM_NAME)
PROBLEM_DEFINITION_FILE = os.path.join(PROBLEM_DIR, f"{PROBLEM_NAME}.py")
PROBLEM_TRAINING_GRID_FILE = os.path.join(PROBLEM_DIR, f"{PROBLEM_NAME}_training_grid.dat")
PROBLEM_DATA_FILE = os.path.join(PROBLEM_DIR, f"{PROBLEM_NAME}_data.dat")
PBS_TEMPLATE_FILE = os.path.join(PROBLEM_DIR, f"{PROBLEM_NAME}-template.pbs")
# print(f"RESEARCH_ROOT = {RESEARCH_ROOT}")
# print(f"PROBLEMS_DIR = {PROBLEMS_DIR}")
# print(f"PROBLEM_DIR = {PROBLEM_DIR}")
# print(f"PROBLEM_DEFINITION_FILE = {PROBLEM_DEFINITION_FILE}")
# print(f"PROBLEM_TRAINING_GRID_FILE = {PROBLEM_TRAINING_GRID_FILE}")
# print(f"PROBLEM_DATA_FILE = {PROBLEM_DATA_FILE}")
# print(f"PBS_TEMPLATE_FILE = {PBS_TEMPLATE_FILE}")

# General constants
MICROSECONDS_PER_SECOND = 1e6

# PBS job control constants
QSUB_COMMAND = "bash"
# QSUB_COMMAND = "qsub"
# PBS_ACCOUNT = "UJHB0019"
# PBS_QUEUE = "main"
# PBS_WALLTIME = "00:05:00"
# PBS_SELECT = "select=1:ncpus=128"

# Run environment constants
RUN_PLATFORM = "ventura"
RUN_PYTHON_ENVIRONMENT = "research-3.10"
RUN_CODE_BRANCH = "waves1d"
RUN_PROBLEM_CLASS = "waves1d"
RUN_PROBLEM_NAME = "eplasma1"

# Problem set constants
N_RUNS = 1
W_MIN = 0.0
W_MAX = 1.0
N_W = 1
WS = np.linspace(W_MIN, W_MAX, N_W)
N_LAYERS = 1
SAVE_MODEL = -1
N_LAYERS = 1
N_HID = 10
MAX_EPOCHS=100

# Read and create the PBS script template.
with open(PBS_TEMPLATE_FILE) as f:
    pbs_template_content = f.read()
pbs_template = Template(pbs_template_content)

# Save the starting directory.
start_dir = os.getcwd()
# print(f"start_dir = {start_dir}")

# Perform all runs.
for w in WS:
    # print(f"w = {w:.2f}")

    # Make a directory for runs using this weight, and go there.
    w_dirname = f"w={w:.2f}"
    # print(f"w_dirname = {w_dirname}")
    os.mkdir(w_dirname)
    os.chdir(w_dirname)
    weight_dir = os.getcwd()
    # print(f"Current directory is {weight_dir}.")

    for run in range(N_RUNS):
        # print(f"run = {run}")

        # Compute the seed as an integer number of microseconds.
        seed = datetime.datetime.now().timestamp()
        seed = int(seed*MICROSECONDS_PER_SECOND)
        # print(f"seed = {seed}")

        # Assemble the run ID string.
        run_id = f"{PROBLEM_NAME}-{run:02d}-{seed}"
        print(f"run_id = {run_id}")

        # Make a directory for this run_id, and go there.
        os.mkdir(run_id)
        os.chdir(run_id)
        cwd = os.getcwd()
        # print(f"Current directory is {cwd}.")

        # Assemble the dictionary for the template.
        options = {
            "run_id": run_id,
            "pbs_jobid": run_id,
            "run_platform": RUN_PLATFORM,
            "run_python_environment": RUN_PYTHON_ENVIRONMENT,
            "run_code_branch": RUN_CODE_BRANCH,
            "run_problem_class": RUN_PROBLEM_CLASS,
            "run_problem_name": RUN_PROBLEM_NAME,
            "max_epochs": MAX_EPOCHS,
            "n_hid": N_HID,
            "n_layers": N_LAYERS,
            "save_model": SAVE_MODEL,
            "seed": seed,
            "w": w,
#             "pbs_account": PBS_ACCOUNT,
#             "pbs_queue": PBS_QUEUE,
#             "pbs_select": PBS_SELECT,
#             "pbs_walltime": PBS_WALLTIME,
        }

        # Render the template for this run and save as a file.
        pbs_content = pbs_template.render(options)
        # print(f"pbs_content = {pbs_content}")
        pbs_file = f"{run_id}.pbs"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)

        # Submit the job and save the output.
        args = [QSUB_COMMAND, pbs_file]
        result = subprocess.run(
            args, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        # Move back to the weight directory.
        os.chdir(weight_dir)
        cwd = os.getcwd()
        # print(f"Current directory is {cwd}.")

    # Move back to the top directory.
    os.chdir(start_dir)
    cwd = os.getcwd()
    # print(f"Current directory is {cwd}.")
