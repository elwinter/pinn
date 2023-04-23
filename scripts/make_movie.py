#!/usr/bin/env python


"""Make a movie of a single variable.

Make a movie of a single variable. The movie is created as an animated GIF.
Individual frames are saved as PNG files in the frames/ subdirectory.
"""


# Import standard modules.
import argparse

# Import 3rd-party modules.

# Import project-specific modules.


# Program constants and defaults

# Program description.
DESCRIPTION = "Make a movie of a single variable."

# Default epoch to use when selecting a trained model.
# -1 = use last epoch in results.
DEFAULT_EPOCH = -1


def create_command_line_parser():
    """Create the command-line argument parser.

    Create the command-line argument parser.

    Parameters
    ----------
    None

    Returns
    -------
    parser : argparse.ArgumentParser
        Command-line argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--epoch", "-e", type=int, default=DEFAULT_EPOCH,
        help="Epoch for trained model to use for movie "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "movie_path", type=str,
        help="Path to directory containing movie data"
    )
    parser.add_argument(
        "problem_name", type=str,
        help="Name of problem used in movie"
    )
    parser.add_argument(
        "movie_type", type=str,
        help="Name of movie type"
    )
    return parser


def main():
    """Main program logic.

    The program starts here.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Set up the command-line parser.
    parser = create_command_line_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    debug = args.debug
    epoch = args.epoch
    verbose = args.verbose
    movie_path = args.movie_path
    problem_name = args.problem_name
    movie_type = args.movie_type
    if debug:
        print(f"args = {args}")
        print(f"debug = {debug}")
        print(f"epoch = {epoch}")
        print(f"verbose = {verbose}")
        print(f"movie_path = {movie_path}")
        print(f"problem_name = {problem_name}")
        print(f"movie_type = {movie_type}")


if __name__ == "__main__":
    """Begin main program."""
    main()
