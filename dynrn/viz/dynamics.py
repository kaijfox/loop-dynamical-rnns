import numpy as np
from matplotlib.collections import LineCollection
from cmap import Colormap
import matplotlib.pyplot as plt


def trajecories(
    ax, x, colors, color="time", set_lim=True, time_point=None, point_kws={}, **kws
):
    """
    Plot array of trajecories colored by time

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x : np.ndarray
        Array of shape (n_traj, n_time, 2) containing the trajecories, or a list
        of arrays of shape (n_time, 2) containing trajectories of variable length.
    colors : np.ndarray
        Array of shape (n_time, 3) containing the color for each time point or
        array of shape (n_traj, 3) containing the color for each trajecory,
        depending on `color` parameter.
    color : str
        If 'time', use the color for each time point, if 'traj', use the color
        for each trajecory
    set_lim : bool
        If True, set the limits of the axes to be the same for both x and y


    """
    # pad and concatenate trajectories of variable length
    if not isinstance(x, np.ndarray):
        max_len = max(len(t) for t in x)
        x = np.array(
            [
                np.pad(
                    t,
                    [(0, max_len - len(t)), (0, 0)],
                    constant_values=(0, np.nan),
                )
                for t in x
            ]
        )
    colors = np.array(colors)

    n_traj = x.shape[0]
    sliding = (
        np.lib.stride_tricks.sliding_window_view(x, 3, axis=1)
        .transpose(0, 1, 3, 2)
        .reshape(-1, 3, 2)
    )
    cdim = colors.shape[-1]
    if color == "time":
        # colors.shape = (colors.n_time, 3/4)
        colors = np.concatenate(
            [colors, colors[[-1] * (x.shape[1] - len(colors))]], axis=0
        )
        colors = colors[: x.shape[1]]
        # colors.shape = (x.n_time, 3/4)
        # c.shape = (n_traj, x.n_time - 2, 3/4)
        c = np.tile(colors[None, 1:-1], [n_traj, 1, 1])
    elif color == "traj":
        # colors.shape = (n_traj, 3/4)
        # c.shape = (n_traj, x.n_time - 2, 3/4)
        c = np.tile(colors[:, None], [1, x.shape[1] - 2, 1])
    elif color == "both":
        # colors.shape = (n_traj, colors.n_time, 3/4)
        colors = np.concatenate(
            [colors, colors[:, [-1] * (x.shape[1] - colors.shape[1])]], axis=1
        )
        # colors.shape = (n_traj, x.n_time, 3/4)
        # c.shape = (n_traj, x.n_time - 2, 3/4)
        c = colors[:, 1:-1]
    c_flat = c.reshape(-1, cdim)
    ax.add_artist(LineCollection(sliding, colors=c_flat, **kws))

    if time_point is not None:
        ax.scatter(x[:, time_point], **{"c": c[:, time_point], **point_kws})

    if set_lim:
        ax.set_aspect(1.0)
        xmin, xmax = np.nanmin(x[..., 0]), np.nanmax(x[..., 0])
        ymin, ymax = np.nanmin(x[..., 1]), np.nanmax(x[..., 1])
        ax.plot([xmin, xmax], [ymin, ymax], alpha=0)


def vectorfield_2d(ax, x, y, u, v, color=None, cmap=None, **kws):
    """
    Wrapper for streamplot with magnitude of vectors as color.
    """
    if color is not None:
        color_kws = {'color': color}
    else:
        velocity = np.sqrt(u**2 + v**2)
        norm = plt.Normalize(velocity.min(), velocity.max())
        if isinstance(cmap, Colormap):
            cmap = cmap.to_mpl()
        if cmap is None:
            cmap = "viridis"
        color_kws = {"color": velocity, "cmap": cmap, "norm": norm}

    kws = {**{"lw": 1, **color_kws}, **kws}

    strm = ax.streamplot(x, y, u, v, color=velocity, linewidth=2, cmap=cmap, **kws)
    return strm

def mapping_2d(ax, x, y, u, v, point = {}, line = {}):
    """
    Plot lines from (x, y) to (x + u, y + v) with points at (u, v) end.
    """
    X, Y = np.meshgrid(x, y)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    u = u.reshape(-1)
    v = v.reshape(-1)
    
    line = {**{"color": "k", "lw": 1}, **line}
    point = {**{"c": "k", "s": 4, 'marker': 's'}, **point}
    
    
    lines = LineCollection(
        np.stack([np.stack([X, X + u], axis=-1), np.stack([Y, Y + v], axis=-1)], axis=-1),
        **line
    )
    ax.add_artist(lines)
    points = ax.scatter(X + u, Y + v, **point)
    
    # invisible points to set limits
    ax.scatter(X, Y, alpha=0)
    ax.set_aspect(1.0)
    return lines, points

def compute_vector_field_2d(f, xrng, yrng, res=40):
    """
    Compute a grid of vectors for a 2D vector field f.

    Parameters
    ----------
    f : callable
        Function taking a 2d numpy array of shape (..., 2,) and returning a 2d
        numpy array of shape (..., 2,) representing the vector field.
    xrng, yrng : tuple
        Ranges for x and y coordinates.
    res : int
        Resolution of the grid.
    """
    x = np.linspace(*xrng, res)
    y = np.linspace(*yrng, res)
    X, Y = np.meshgrid(x, y)
    X_new, Y_new = np.moveaxis(f(np.stack([X, Y], axis=-1)), -1, 0)
    U = X_new - X
    V = Y_new - Y
    return x, y, U, V