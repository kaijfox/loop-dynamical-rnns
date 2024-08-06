import numpy as np
import numpy.linalg as la

def is_converging(histories, window = 20, thresh = 1e-2, eps = 1e-4):
    """
    histories : (..., t, N)
    Returns
    -------
    is_converging : bool (...,)

    """
    histories = histories[..., -window:, :]
    magnitudes = la.norm(np.diff(histories, axis = -2), axis = -1)
    decreasing = np.all(np.diff(magnitudes) <= 0, axis = -1)
    small = magnitudes.mean(axis = -1) < thresh
    fixed = magnitudes.mean(axis = -1) < eps
    return (small & decreasing) | fixed


def velocities(histories):
    return la.norm(np.diff(histories, axis = -2), axis = -1)