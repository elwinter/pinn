#!/usr/bin/env python

import os
# import subprocess

from jinja2 import Template

# Define the range of random number seeds to use.
seeds = list(range(5, 6))
print(f"seeds = {seeds}")

# Define the PINN code root.
PINN_ROOT = os.path.join(
    os.environ["HOME"], "research_local", "src", "pinn"
)
print(f"PINN_ROOT = {PINN_ROOT}")

# Specify the branch path.
BRANCH = "periodic_save_model"
BRANCH_PATH = os.path.join(PINN_ROOT, BRANCH, "pinn")
print(f"BRANCH_PATH = {BRANCH_PATH}")

# Define problem location.
PROBLEM_CLASS = "loop2d"
PROBLEM_NAME = "loop2d_BxBy"
PROBLEM_ROOT = os.path.join(
    BRANCH_PATH, "problems", PROBLEM_CLASS, PROBLEM_NAME
)
print(f"PROBLEM_ROOT = {PROBLEM_ROOT}")

# Define PINN command.
PINN_CMD = os.path.join(BRANCH_PATH, "pinn", "pinn1.py")
print(f"PINN_CMD = {PINN_CMD}")

# Specify the command template.
CMD_TEMPLATE = (
    "{{ pinn_cmd }}"
    " --debug"
    " --verbose"
    " --seed={{ seed }}"
    " --max_epochs={{ max_epochs }}"
    " --save_model={{ save_model }}"
    " --n_layers={{ n_layers }}"
    " --n_hid={{ n_hid }}"
    " --data={{ data_path }}"
    " -w={{ w }}"
    " {{ problem_path }}"
    " {{ training_points_path }}"
)
print(f"CMD_TEMPLATE = {CMD_TEMPLATE}")
cmd_template = Template(CMD_TEMPLATE)

# Specify problem files.
data_path = os.path.join(
    PROBLEM_ROOT, f"{PROBLEM_NAME}_initial_conditions.dat"
)
problem_path = os.path.join(
    PROBLEM_ROOT, f"{PROBLEM_NAME}.py"
)
training_points_path = os.path.join(
    PROBLEM_ROOT, f"{PROBLEM_NAME}_training_grid.dat"
)

# Specify standard options for the set.
options = {
    "pinn_cmd": PINN_CMD,
    "max_epochs": 100,
    "save_model": 50,
    "n_layers": 4,
    "n_hid": 100,
    "data": data_path,
    "w": 0.95,
    "problem_path": problem_path,
    "training_points_path": training_points_path,
}

original_cwd = os.getcwd()

for s in seeds:
    print("==========")
    print(f"Performing run for seed = {s}")
    run_path = str(s)
    os.mkdir(run_path)
    options["seed"] = s
    cmd = cmd_template.render(options)
    print(f"cmd = {cmd}")
    os.chdir(run_path)
    with open("cmd", "w") as f:
        f.write(cmd)
    os.system(cmd)
    os.chdir(original_cwd)
