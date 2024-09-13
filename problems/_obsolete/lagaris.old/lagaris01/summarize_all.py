import glob
import importlib
import os
import re
import sys

import numpy as np


# Run environment constants
RUN_PROBLEM_NAME = "lagaris01"

# Problem set constants
N_RUNS = 3
W_MIN = 0.0
W_MAX = 1.0
N_W = 3
WS = np.linspace(W_MIN, W_MAX, N_W)

# Save the starting directory.
start_dir = os.getcwd()

# Summarize all runs.
for w in WS:

    # Compute the weight directory and go there..
    w_dirname = f"w={w:.2f}"
    os.chdir(w_dirname)

    # Examine each run subdirectory for this weight.
    for run in range(N_RUNS):

        # Assemble the runid glob pattern.
        runid_glob = f"{RUN_PROBLEM_NAME}-{run:02d}-*"

        # Compute the runid.
        runid = glob.glob(runid_glob)[0]

        # Enter the run directory.
        os.chdir(runid)

        # Compute the name of the run log.
        run_log_glob = f"{runid}.o*"
        run_log_name = glob.glob(run_log_glob)[0]

        # Read the run log.
        with open(run_log_name, "r") as f:
            lines = f.readlines()

        # Extract the run end time.
        line_pattern = "Training stopped at (.+)$"
        for line in reversed(lines):
            m = re.match(line_pattern, line)
            if m:
                break
        run_end_time = m.groups()[0]

        # Extract the name of the HPC system.
        line_pattern = "hpc_system=(.+)"
        for line in lines:
            m = re.match(line_pattern, line)
            if m:
                break
        platform = m.groups()[0]

        # Extract the total training time.
        line_pattern = "Total training time was (.+) seconds."
        for line in reversed(lines):
            m = re.match(line_pattern, line)
            if m:
                break
        run_time = m.groups()[0]

        # Extract the final values of the loss functions.
        line_pattern = "Ending epoch \d+, \(L, L_res, L_data\) = \((.+), (.+), (.+)\)"
        for line in reversed(lines):
            m = re.match(line_pattern, line)
            if m:
                break
        (L, L_all, L_bc) = m.groups()[:]

        # Read the hyperparameter file and execute it.
        # I have to do it this way since I can't get a reimport to work on each pass.
        hyperparameter_path = os.path.join(".", RUN_PROBLEM_NAME, "hyperparameters.py")
        with open(hyperparameter_path, "r") as f:
            lines = f.readlines()
        hyperparameter_commands = "\n".join(lines)
        exec(hyperparameter_commands)

        # Print a summary line for this run.
        summary = ",".join([
            run_end_time,
            platform,
            precision,
            str(n_layers),
            str(n_hid),
            str(learning_rate),
            str(w_data),
            str(max_epochs),
            run_time,
            L, L_all, L_bc
        ])
        print(summary)

        # Move back to the weight directory.
        os.chdir("..")

    # Move back to the starting directory.
    os.chdir(start_dir)
