import datetime
import os
import random
import subprocess

from jinja2 import Template
import numpy as np


# General constants
MICROSECONDS_PER_SECOND = 1e6

# PBS job control constants
QSUB_COMMAND = "qsub"
PBS_TEMPLATE_FILE = "linecurrent_BxBy-template.pbs"
PBS_ACCOUNT = "UJHB0019"
PBS_QUEUE = "main"
PBS_WALLTIME = "00:30:00"
PBS_SELECT = "select=1:ncpus=128"

# Run environment constants
RUN_HPC_SYSTEM = "derecho"
RUN_PYTHON_ENVIRONMENT = "research-3.10"
RUN_CODE_BRANCH = "development"
RUN_PROBLEM_CLASS = "linecurrent"
RUN_PROBLEM_NAME = "linecurrent_BxBy"

# Problem set constants
N_RUNS = 3
W_MIN = 0.0
W_MAX = 1.0
N_W = 3
WS = np.linspace(W_MIN, W_MAX, N_W)
N_LAYERS = 1
SAVE_MODEL = 1000
N_LAYERS = 4
N_HID = 100
MAX_EPOCHS=10000

# Read and create the PBS script template.
with open(PBS_TEMPLATE_FILE) as f:
    pbs_template_content = f.read()
pbs_template = Template(pbs_template_content)

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
    cwd = os.getcwd()
    print(f"Current directory is {cwd}.")

    for run in range(N_RUNS):
        print(f"run = {run}")

        # Compute the seed as an integer number of microseconds.
        seed = datetime.datetime.now().timestamp()
        seed = int(seed*MICROSECONDS_PER_SECOND)
        print(f"seed = {seed}")

        # Assemble the run ID string.
        runid = f"{RUN_PROBLEM_NAME}-{run:02d}-{w:.2f}-{seed}"
        print(f"runid = {runid}")

        # Make a directory for this runid, and go there.
        os.mkdir(runid)
        os.chdir(runid)
        cwd = os.getcwd()
        print(f"Current directory is {cwd}.")

        # Assemble the dictionary for the template.
        options = {
            "run_hpc_system": RUN_HPC_SYSTEM,
            "run_python_environment": RUN_PYTHON_ENVIRONMENT,
            "run_code_branch": RUN_CODE_BRANCH,
            "run_problem_name": RUN_PROBLEM_NAME,
            "run_problem_class": RUN_PROBLEM_CLASS,
            "pbs_runid": runid,
            "pbs_account": PBS_ACCOUNT,
            "pbs_queue": PBS_QUEUE,
            "pbs_select": PBS_SELECT,
            "pbs_walltime": PBS_WALLTIME,
            "max_epochs": MAX_EPOCHS,
            "n_hid": N_HID,
            "n_layers": N_LAYERS,
            "save_model": SAVE_MODEL,
            "seed": seed,
            "w": w,
        }

        # Render the template for this run and save as a file.
        pbs_content = pbs_template.render(options)
        # print(f"pbs_content = {pbs_content}")
        pbs_file = f"{runid}.pbs"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)

        # Submit the job.
        args = [QSUB_COMMAND, pbs_file]
        subprocess.run(
            args, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        # Move back to the weight directory.
        os.chdir("..")
        cwd = os.getcwd()
        print(f"Current directory is {cwd}.")

    # Move back to the top directory.
    os.chdir("..")
    cwd = os.getcwd()
    print(f"Current directory is {cwd}.")
