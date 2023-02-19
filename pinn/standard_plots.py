"""Routines for standard plots.

This module provides functions which create standard plots.

Author
------
Eric Winter (eric.winter62@gmail.com)

"""


# Import standard modules.

# Import supplemental modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import project modules.


# Module constants

# Default minimum and maximum loss function values to plot
L_MIN_DEFAULT = 1e-9
L_MAX_DEFAULT = 1


def plot_loss_functions(
    Ls, labels=None,
    ax=None, title="Loss functions",
    vmin=L_MIN_DEFAULT, vmax=L_MAX_DEFAULT,
    show_xlabel=True, show_ylabel=True
):
    """Plot one or more loss functions.

    Plot one or more loss functions as a function of training epoch.

    Parameters
    ----------
    Ls : list of np.ndarray, length n, each shape (n,)
        Loss function values to plot.
    labels : list of str, length n, default None
        Label strings for each loss function.
    ax : matplotlib.Axes, default None
        Axes object to use for the plot.
    title : str, default "Loss functions"
        Title for plot
    vmin : float, default L_MIN_DEFAULT
        Minimum value for loss function vertical axis.
    vmax : float, default L_MAX_DEFAULT
        Maximum value for loss function vertical axis.
    show_xlabel : bool, default True
        Set to True to show label for horizontal axis.
    show_ylabel : bool, default True
        Set to True to show label for vertical axis.

    Returns
    -------
    None
    """
    # If no labels were provided, generate numeric labels.
    if labels is None:
        labels = [str(i) for i in range(len(L))]

    # If no axis was specified, get the current axis.
    if ax is None:
        ax = plt.gca()

    # Plot each loss history on a logarithmic vertical scale.
    for (L, label) in zip(Ls, labels):
        ax.semilogy(L, label=label)

    # Decorate the plot.
    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel("Epoch")
    if show_ylabel:
        ax.set_ylabel("Loss function")
    ax.set_ylim(vmin, vmax)
    ax.grid()
    ax.legend()


def plot_BxBy_quiver(
    x, y, Bx, By,
    ax=None, title="Magnetic field",
    show_xlabel=True, show_ylabel=True,
    x_tick_pos=None, x_tick_labels=None,
    y_tick_pos=None, y_tick_labels=None
):
    """Plot Bx and By components as a quiver plot.

    Plot Bx and By components as a quiver plot.

    Parameters
    ----------
    x, y : np.ndarray, shape (n,)
        x and y coordinates for arrows.
    Bx, By : np.ndarray, shape (n,)
        x- and y-components for arrows.
    ax : matplotlib.Axes, default None
        Axes object to use for the plot.
    title : str, default "Magnetic field"
        Title for plot
    show_xlabel : bool, default True
        Set to True to show label for horizontal axis.
    show_ylabel : bool, default True
        Set to True to show label for vertical axis.
    x_tick_pos : np.ndarray of float, default None
        Positions in data coordinates for horizontal axis tick marks.
    x_tick_labels : list of str, same length as x_tick_pos, default None
        Strings for tick mark labels on horizontal axis.
    y_tick_pos : np.ndarray of float, default None
        Positions in data coordinates for vertical axis tick marks.
    y_tick_labels : list of str, same length as y_tick_pos, default None
        Strings for tick mark labels on vertical axis.

    Returns
    -------
    None
    """
    # If no axis was specified, get the current axis.
    if ax is None:
        ax = plt.gca()

    # Plot the data as arrows.
    ax.quiver(x, y, Bx, By)

    # Decorate the plot.
    ax.set_aspect(1.0)
    if show_xlabel:
        ax.set_xlabel("$x$")
    if show_ylabel:
        ax.set_ylabel("$y$")
    if x_tick_pos is not None and x_tick_labels is not None:
        ax.set_xticks(x_tick_pos, x_tick_labels)
    if y_tick_pos is not None and y_tick_labels is not None:
        ax.set_yticks(y_tick_pos, y_tick_labels)
    ax.grid()
    ax.set_title(title)


def plot_logarithmic_heatmap(
    z,
    ax=None, title="Logarithmic heat map",
    vmin=None, vmax=None,
    show_xlabel=True, show_ylabel=True,
    x_tick_pos=None, x_tick_labels=None,
    y_tick_pos=None, y_tick_labels=None
):
    """Plot a logarithmic heat map.

    Plot a logarithmic heat map.

    Parameters
    ----------
    z : np.ndarray, shape (n,)
        Values for heatmap.
    ax : matplotlib.Axes, default None
        Axes object to use for the plot.
    title : str, default "Logarithmic heat map"
        Title for plot
    vmin : float, default None
        Minimum mapped value for plot.
    vmax : float, default None
        Maximum mapped value for plot.
    show_xlabel : bool, default True
        Set to True to show label for horizontal axis.
    show_ylabel : bool, default True
        Set to True to show label for vertical axis.
    x_tick_pos : np.ndarray of float, default None
        Positions in data coordinates for horizontal axis tick marks.
    x_tick_labels : list of str, same length as x_tick_pos, default None
        Strings for tick mark labels on horizontal axis.
    y_tick_pos : np.ndarray of float, default None
        Positions in data coordinates for vertical axis tick marks.
    y_tick_labels : list of str, same length as y_tick_pos, default None
        Strings for tick mark labels on vertical axis.

    Returns
    -------
    None
    """
    # If no axis was specified, get the current axis.
    if ax is None:
        ax = plt.gca()

    # Plot the data as a logarithmic heat map.
    sns.heatmap(z, ax=ax, norm=mpl.colors.LogNorm(), square=True,
                vmin=vmin, vmax=vmax)

    # Decorate the plot.
    if show_xlabel:
        ax.set_xlabel("$x$")
    if show_ylabel:
        ax.set_ylabel("$y$")
    if x_tick_pos is not None and x_tick_labels is not None:
        ax.set_xticks(x_tick_pos, x_tick_labels, rotation=0)
    if y_tick_pos is not None and y_tick_labels is not None:
        ax.set_yticks(y_tick_pos, y_tick_labels)
    ax.grid()
    ax.set_title(title)


def plot_linear_heatmap(
    z,
    ax=None, title="Linear heat map",
    vmin=None, vmax=None,
    show_xlabel=True, show_ylabel=True,
    x_tick_pos=None, x_tick_labels=None,
    y_tick_pos=None, y_tick_labels=None
):
    """Plot a linear heat map.

    Plot a linear heat map.

    Parameters
    ----------
    z : np.ndarray, shape (n,)
        Values for heatmap.
    ax : matplotlib.Axes, default None
        Axes object to use for the plot.
    title : str, default "Linear heat map"
        Title for plot
    vmin : float, default None
        Minimum mapped value for plot.
    vmax : float, default None
        Maximum mapped value for plot.
    show_xlabel : bool, default True
        Set to True to show label for horizontal axis.
    show_ylabel : bool, default True
        Set to True to show label for vertical axis.
    x_tick_pos : np.ndarray of float, default None
        Positions in data coordinates for horizontal axis tick marks.
    x_tick_labels : list of str, same length as x_tick_pos, default None
        Strings for tick mark labels on horizontal axis.
    y_tick_pos : np.ndarray of float, default None
        Positions in data coordinates for vertical axis tick marks.
    y_tick_labels : list of str, same length as y_tick_pos, default None
        Strings for tick mark labels on vertical axis.

    Returns
    -------
    None
    """
    # If no axis was specified, get the current axis.
    if ax is None:
        ax = plt.gca()

    # Plot the data as a linear heat map.
    sns.heatmap(z, ax=ax, square=True, vmin=vmin, vmax=vmax)

    # Decorate the plot.
    if show_xlabel:
        ax.set_xlabel("$x$")
    if show_ylabel:
        ax.set_ylabel("$y$")
    if x_tick_pos is not None and x_tick_labels is not None:
        ax.set_xticks(x_tick_pos, x_tick_labels, rotation=0)
    if y_tick_pos is not None and y_tick_labels is not None:
        ax.set_yticks(y_tick_pos, y_tick_labels)
    ax.grid()
    ax.set_title(title)
