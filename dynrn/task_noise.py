import torch as th
import numpy as np
import numpy.linalg as la



def fftnoise(f, rng):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = rng.uniform(size = Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real


def lowpass_noise(max_freq, samples=1024, samplerate=1, size = (), seed = 0):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    rng = np.random.default_rng(seed)

    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=0, freqs<=max_freq))[0]
    f[idx] = 1

    ret = np.zeros(size + (samples,))
    for i in np.ndindex(size):
        ret[i] = fftnoise(f, rng)
    return ret


def normalize_range(x, limits, axis = None):
    """
    Normalize x to the range of y independently for each index along axis.

    Parameters
    ----------
    x : np.ndarray (..., N, ...)
        The data to normalize.
    limits : Tuple of arrays, shape (N,)
        The tatget min and max for each element of axis.
    axis : int, or tuple[int], optional
        The axis along which to measure max and min. The default is None.
    """
    if isinstance(axis, int):
        axis = (axis,)
    if axis is None:
        axis = tuple()
    axis = tuple(x.ndim + i if i < 0 else i for i in axis)

    over_ax = tuple(i for i in range(x.ndim) if i not in axis)
    expand = tuple((None if i in over_ax else Ellipsis) for i in range(x.ndim))
    
    xmin = x.min(axis = over_ax, keepdims=True)
    xmax = x.max(axis = over_ax, keepdims=True)
    normed = (x - xmin) / (xmax - xmin)
    vmin, vmax = limits[0][expand], limits[1][expand]
    return (normed * (vmax - vmin) + vmin)


def get_limits(x, axis = None):
    """
    Calculate the range of x for each index along axis.

    Parameters
    ----------
    x : np.ndarray
        The data to calculate the range of.
    axis : int, or tuple[int], optional
        The axis along which to measure max and min. The default is None.
    Returns
    -------
    range : tuple[np.ndarray]
        The min and max of x along axis
    """
    if isinstance(axis, int):
        axis = (axis,)
    if axis is None:
        axis = tuple()
    axis = tuple(x.ndim + i if i < 0 else i for i in axis)

    over_ax = tuple(i for i in range(x.ndim) if i not in axis)
    return (x.min(axis = over_ax), x.max(axis = over_ax))


def normalize_stats(x, stats, axis = None):
    """
    Normalize x to target statistics independently for each index along axis.

    Parameters
    ----------
    x : np.ndarray (..., N, ...)
        The data to normalize.
    limits : Tuple of arrays, shape (N,)
        The tatget min and max for each element of axis.
    axis : int, or tuple[int], optional
        The axis along which to measure max and min. The default is None.
    """
    if isinstance(axis, int):
        axis = (axis,)
    if axis is None:
        axis = tuple()
    axis = tuple(x.ndim + i if i < 0 else i for i in axis)

    over_ax = tuple(i for i in range(x.ndim) if i not in axis)
    expand = tuple((None if i in over_ax else Ellipsis) for i in range(x.ndim))
    
    xmean = x.mean(axis = over_ax, keepdims=True)
    xstd = x.std(axis = over_ax, keepdims=True)
    normed = (x - xmean) / xstd
    vmean, vstd = stats[0][expand], stats[1][expand]
    return (normed * vstd + vmean)


def get_stats(x, axis = None):
    """
    Calculate the first and second order statistic of x for each index along
    axis.

    Parameters
    ----------
    x : np.ndarray
        The data to calculate the range of.
    axis : int, or tuple[int], optional
        The axis along which to measure stats. The default is None.
    Returns
    -------
    range : tuple[np.ndarray]
        The statistics of x along axis
    """
    if isinstance(axis, int):
        axis = (axis,)
    if axis is None:
        axis = tuple()
    axis = tuple(x.ndim + i if i < 0 else i for i in axis)

    over_ax = tuple(i for i in range(x.ndim) if i not in axis)
    return (x.mean(axis = over_ax), x.std(axis = over_ax))