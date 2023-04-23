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

# Import 3rd-party modules.
from jinja2 import Template

# Import project modules.


# Program constants

# Program description string for help text.
DESCRIPTION = "Make sets of movies for a PINN seed set."


# Define the PINN code root.
PINN_ROOT = os.environ["RESEARCH_INSTALL_DIR"]

# Define the movie-making command.
MOVIE_SCRIPT = "make_movie.py"
MOVIE_CMD = os.path.join(PINN_ROOT, "pinn", "scripts", MOVIE_SCRIPT)

# Define the jinja2 command template.
CMD_TEMPLATE = (
    "{{ movie_script }}"
    " {{ debug }}"
    " {{ verbose }}"
    " >> {{ movie_script }}.out"
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
        "--debug", "-d", action="store_true", default=False,
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
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
    verbose = args.verbose
    if debug:
        print(f"args = {args}")
        print(f"debug = {debug}")
        print(f"verbose = {verbose}")

    # Create the command template.
    cmd_template = Template(CMD_TEMPLATE)
    if debug:
        print(f"cmd_template = {cmd_template}")

    # Assemble the options dictionary.
    options = {}
    if debug:
        options["debug"] = "--debug"
    if verbose:
        options["verbose"] = "--verbose"

    # Create the movie sets for each seed.
    for seed in seeds:
        print("==========")
        print(f"Making movies for seed = {seed}")

        # Render the template to create the command string.
        cmd = cmd_template.render(options)
        if debug:
            print(f"cmd = {cmd}")

        # Run the command.
        os.system(cmd)


if __name__ == "__main__":
    """Begin main program."""
    main()
