"""Routines for creating neural network training data.

This module provides functions which perform common calculations for
creating training data for neural networks.

Author
------
Eric Winter (eric.winter62@gmail.com)

"""


# Import standard modules.

# Import supplemental modules.
import numpy as np

# Import proejct modules


def create_training_points_gridded(ng: np.ndarray, bg: np.ndarray):
    """Create evenly-spaced training points.

    Create a set of training points evenly spaced in n orthogonal dimensions.
    Flatten the data to a list of of n-dimensional points.

    Parameters
    ----------
    ng : np.ndarray of int, shape (n_dim,)
        Number of evenly-space grid points along each dimension.
    bg : np.ndarray of float, shape (n_dim, 2)
        Minimum and maximum grid values along each training dimension.

    Returns
    -------
    Xg : np.ndarray, shape (np.prod(ng), n_dim)
        Array of all training points.
    """
    # Determine the number of dimensions.
    n_dim = len(ng)

    # Compute the grid points for the each dimension.
    xg = []
    for i in range(n_dim):
        xg.append(np.linspace(bg[i][0], bg[i][1], ng[i]))

    # Now build the grid one dimension at a time.
    grid = []
    for i in range(n_dim):
        grid.append(
            np.tile(
                np.repeat(xg[i], np.prod(ng[i + 1:])),
                np.prod(ng[:i])
            )
        )
    Xg = np.vstack(grid).T

    # Flatten the list of training points.
    Xg.shape = (np.prod(ng), n_dim)

    # Return the training grid.
    return Xg


def create_training_points_random(n: int, b: np.ndarray):
    """Create randomly-spaced training points.

    Create a set of training points randomly spaced in n orthogonal dimensions.
    Return the data as a ndarray of shape (n, n_dim).

    Parameters
    ----------
    n : int
        Number of randomly-spaceed points to create in the domaim.
    b : np.ndarray of float, shape (n_dim, 2)
        Minimum and maximum (boundary) values for each dimension.

    Returns
    -------
    X : np.ndarray, shape (n, n_dim)
        Array of all training points.
    """
    # Determine the number of dimensions.
    n_dim = b.shape[0]

    # Create n random points in each dimension.
    x_random = []
    for i in range(n_dim):
        x = np.random.uniform(low=b[i][0], high=b[i][1], size=n)
        x_random.append(x)

    # Stack the vectors to get a single array of shape (n, n_dim).
    X = np.vstack(x_random).T

    # Return the training points.
    return X


if __name__ == '__main__':

    # Gridded training points.
    ng = np.array([2, 3, 4, 5], dtype=int)
    print("ng = %s" % ng)
    bg = np.array([
        [0, 1],
        [1, 2],
        [3, 4],
        [3.5, 5.5],
    ])
    print("bg = %s" % bg)
    Xg = create_training_points_gridded(ng, bg)
    print("Xg = %s" % Xg)

    # Random training points.
    nr = ng
    Nr = np.prod(nr)
    br = bg
    n_dim = len(br)
    Xr = create_training_points_random(nr, br)
    print(Xr)
