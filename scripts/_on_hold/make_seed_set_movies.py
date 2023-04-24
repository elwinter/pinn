#!/usr/bin/env python


"""Make sets of movies for a PINN seed set.

Make sets of movies for a PINN seed set. A "seed set" in this case means the
same problem, trained multiple times, using the same network and training
hyperparameters, each time using a different random number seed.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.
import argparse
import os
import re

# Import 3rd-party modules.
from jinja2 import Template

# Import project modules.


# Program constants

# Program description string for help text.
DESCRIPTION = "Make sets of movies for a PINN seed set."

# Default list of movies to make.
DEFAULT_MOVIES = "B"


# Define the PINN code root.
PINN_ROOT = os.environ["RESEARCH_INSTALL_DIR"]

# Define the movie-making command.
MOVIE_SCRIPT = "make_movie.py"
MOVIE_SCRIPT_NAME = MOVIE_SCRIPT.rstrip(".py")
MOVIE_CMD = os.path.join(PINN_ROOT, "pinn", "scripts", MOVIE_SCRIPT)

# Define the jinja2 command template.
CMD_TEMPLATE = (
    "{{ movie_cmd }}"
    " {{ debug }}"
    " {{ verbose }}"
    " >> {{ movie_script_name }}.out"
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
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--movies", type=str, default=DEFAULT_MOVIES,
        help="Movies to make (comma-separated names)"
             " (default: %(default)s)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "seed_set_path", type=str,
        help="Path to directory containing seed set"
    )
    parser.add_argument(
        "problem_name", type=str,
        help="Name of problem used in seed set"
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
    movies_str = args.movies
    verbose = args.verbose
    seed_set_path = args.seed_set_path
    problem_name = args.problem_name
    if debug:
        print(f"args = {args}")
        print(f"debug = {debug}")
        print(f"movies_str = {movies_str}")
        print(f"verbose = {verbose}")
        print(f"seed_set_path = {seed_set_path}")
        print(f"problem_name = {problem_name}")

    # Split the movie types string into a list.
    movie_types = movies_str.split(",")
    if debug:
        print(f"movie_types = {movie_types}")

    # Create the command template.
    cmd_template = Template(CMD_TEMPLATE)
    if debug:
        print(f"cmd_template = {cmd_template}")

    # Assemble the options dictionary.
    options = {}
    if debug:
        options["debug"] = "--debug"
    options["movie_cmd"] = MOVIE_CMD
    options["movie_script"] = MOVIE_SCRIPT
    options["movie_script_name"] = MOVIE_SCRIPT_NAME
    if verbose:
        options["verbose"] = "--verbose"

    # Move to the seed set directory.
    os.chdir(seed_set_path)

    # Make a list of the availble seeds.
    subdirs = list(os.walk("."))[0][1]
    seeds = [int(s) for s in subdirs if re.match("^\d+$", s)]
    seeds.sort()
    if debug:
        print(f"seeds = {seeds}")

    # Create the movie sets for each seed.
    for seed in seeds:
        print("==========")
        print(f"Making movies for seed = {seed}")

        # Render the template to create the command string.
        cmd = cmd_template.render(options)
        if debug:
            print(f"cmd = {cmd}")

        # # Run the command.
        # os.system(cmd)


if __name__ == "__main__":
    """Begin main program."""
    main()
