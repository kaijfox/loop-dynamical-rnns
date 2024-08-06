import numpy as np
import scipy.stats
import torch as th
from mplutil import util as vu
import pandas as pd
from collections import defaultdict
from cmap import Colormap

from .viz.styles import getc

class SimpleTasks:

    @staticmethod
    def fixed_interval(
        interval,
        iti_halflife=5,
        iti_min=3,
        session_length=30,
        n_dim=2,
        n_sessions=2,
        seed=0,
        states=False,
    ):
        """
        Targets as replicas of one-hot inputs delayed by a fixed interval

        Parameters
        ----------
        interval : int
            Number of steps between input and target
        iti_halflife : int
            Half life of the ITI exponential distribution in steps
        iti_min : int
            Minimum ITI in steps
        session_length : int
            Number of steps in a session
        n_dim : int
            Number of dimensions in the input and output
        n_sessions : int
            Number of sessions to generate
        seed : int
            Random seed

        Returns
        -------
        inputs : array (n_sessions, session_length, n_dim)
            Inputs for each session
        targets : array (n_sessions, session_length, n_dim)
            Targets for each session
        states : array (n_sessions, session_length, 3 + iti_min)
            One-hot state vectors for each session. The the last dimension index
            states as
            - 0: ITI exponential countdown
            - 1 to iti_min: ITI deterministic countdown
            For i in range n_dim
            - (iti_min) + (i * (interval + 1)): Input i
            - (iti_min) + (i * (interval + 1)) + (1 to interval): Interval
            countdown i
            - (iti_min) + (i * (interval + 1)) + (interval + 1): Reward i


        """
        inputs = np.zeros((n_sessions, session_length, n_dim))
        targets = np.zeros((n_sessions, session_length, n_dim))
        if states:
            states_ = np.zeros((n_sessions, session_length), dtype=int)
        rng = np.random.default_rng(seed)

        for i in range(n_sessions):
            x = 0
            while True:
                # select timing and dimension for next pulse
                iti = rng.exponential(iti_halflife / np.log(2))
                next_x = x + int(np.ceil(iti)) + iti_min - 1
                if next_x + interval >= session_length:
                    break
                ix = rng.integers(n_dim)
                # set input and target for pulse add states for the trial
                inputs[i, next_x, ix] = 1
                targets[i, next_x + interval, ix] = 1
                if states:
                    ix_ofs = iti_min + ix * (interval + 1)
                    states_[i, x + 1 : x + iti_min] = np.arange(iti_min - 1) + 1
                    states_[i, x + iti_min : next_x] = 0
                    states_[i, next_x] = ix_ofs
                    states_[i, next_x : next_x + interval] = (
                        np.arange(interval) + ix_ofs
                    )
                    states_[i, next_x + interval] = ix_ofs + interval
                x = next_x + interval

        # convert integer states to one-hot and return
        if states:
            nstates = iti_min + n_dim * (interval + 2)
            states = np.zeros((n_sessions, session_length, nstates))
            for i in range(n_sessions):
                states[i, np.arange(session_length), states_[i]] = 1
            return inputs, targets, states
        return inputs, targets

    @staticmethod
    def cdfi(
        interval,
        flip_halflife=12,
        flip_min=5,
        iti_halflife=5,
        iti_min=1,
        session_length=30,
        n_dim=2,
        n_sessions=2,
        seed=0,
    ):
        """
        Fixed interval task with a context switch
        """
        inp, tgt = SimpleTasks.fixed_interval(
            interval, iti_halflife, iti_min, session_length, n_dim, n_sessions, seed
        )
        rng = np.random.default_rng(seed + 1)
        context = np.zeros((n_sessions, session_length, n_dim))
        for i in range(n_sessions):
            x = 0
            ix = rng.integers(n_dim)
            while True:
                flip = rng.exponential(flip_halflife / np.log(2))
                next_x = x + int(np.ceil(flip)) + flip_min - 1
                new_ix = rng.integers(n_dim - 1)
                ix = new_ix if new_ix < ix else new_ix + 1
                if next_x >= session_length:
                    context[i, x:, ix] = 1
                    break
                context[i, x:next_x, ix] = 1
                x = next_x
        tgt = tgt * context
        inp = np.concatenate([inp, context], axis=-1)
        return inp, tgt

    @staticmethod
    def ohflip(halflife, tmin, ndim=2, session_length=30, n_sessions=2, seed=0):
        """
        One-hot flip task.

        With Poisson-timed events of one-hot inputs, maintain activity in the
        corresponding output dimension until the next event.

        Parameters
        ----------
        interval : int
            Number of steps between input and target
        halflife : int
            Half life the exponential defining event timing above `tmin`.
        tmin : int
            Minimum time between events.
        ndim : int
            Number of dimensions in the input and output
        session_length : int
            Number of steps in a session
        n_sessions : int
            Number of sessions to generate
        seed : int
            Random seed

        Returns
        -------
        inputs : array (n_sessions, session_length, ndim)
            Inputs for each session
        targets : array (n_sessions, session_length, ndim)
            Targets for each session
        """
        inp = np.zeros((n_sessions, session_length, ndim))
        tgt = np.zeros((n_sessions, session_length, ndim))
        rng = np.random.default_rng(seed)

        ### generate events, setting a random index of inp to 1 at each event
        #   and setting the corresponding index of tgt to 1 until the next event
        for i in range(n_sessions):
            x = 0
            ix = rng.integers(ndim)
            while True:
                iti = rng.exponential(halflife / np.log(2))
                x += int(np.ceil(iti)) + tmin - 1
                if x >= session_length:
                    break
                new_ix = rng.integers(ndim - 1)
                ix = new_ix if new_ix < ix else new_ix + 1
                inp[i, x, ix] = 1
                tgt[i, x:, :] = 0
                tgt[i, x:, ix] = 1

        return inp, tgt


class itiexp(scipy.stats.rv_continuous):

    def __init__(self, halflife, tmin):
        super().__init__()
        self.exp = scipy.stats.expon(scale=halflife / np.log(2))
        self.tmin = tmin

    def _pdf(self, x, *a, **kw):
        return self.exp.pdf(x - self.tmin, *a, **kw)

    def _rvs(self, *a, **kw):
        return self.exp.rvs(*a, **kw) + self.tmin


class DriscollTasks:

    @staticmethod
    def memorypro(
        iti=itiexp(6, 4),
        context=itiexp(5, 2),
        stim=itiexp(2, 1),
        memory=itiexp(6, 4),
        response=itiexp(5, 2),
        magnitude=scipy.stats.uniform(0.5, 1),
        angle=scipy.stats.uniform(0, np.pi / 2),
        angle_noise=scipy.stats.norm(0, 0.0),
        flag_noise=scipy.stats.norm(0, 0.1),
        session_length=30,
        n_sessions=2,
        seed=0,
    ):
        """
        Memory pro task from Driscoll et al 2022

        Parameters
        ----------
        interval : float
            The time interval between stimuli presentation in seconds.
        iti_halflife : float
            The half-life of the exponential distribution used to generate the
            inter-trial intervals (ITIs) in seconds.
        iti_min : float, optional
            The minimum ITI duration in seconds. Default is 3.
        session_length : int, optional
            The duration of each session in minutes. Default is 30.
        n_sessions : int, optional
            The number of sessions to run. Default is 2.
        seed : int, optional
            The random seed for reproducibility. Default is 0.

        Returns
        -------
        inputs : array (n_sessions, session_length, 3)
            The input stimuli for each session, with the last dimension indexed
            as [fixation, response, stimulus_cos, stimulus_sin]
        targets : array (n_sessions, session_length, 3)
            The target stimuli for each session, with the last dimension indexed
            as [response_cos, response_sin]
        period : array (n_sessions, session_length)
            The period in a trial of each timepoint, with integers 0 to 4
            indicating [iti, context, stim, memory, response], respectively.
        """
        inputs = np.zeros((n_sessions, session_length, 4))
        targets = np.zeros((n_sessions, session_length, 2))
        periods = np.zeros((n_sessions, session_length))
        rng = np.random.default_rng(seed)

        for i in range(n_sessions):
            t = 0
            while True:
                times = t + np.cumsum(
                    [
                        0,
                        iti.rvs(random_state=rng).astype("int"),
                        context.rvs(random_state=rng).astype("int"),
                        stim.rvs(random_state=rng).astype("int"),
                        memory.rvs(random_state=rng).astype("int"),
                        response.rvs(random_state=rng).astype("int"),
                    ]
                )
                if times[-1] >= session_length:
                    break

                theta = angle.rvs(random_state=rng)
                A = magnitude.rvs(random_state=rng)

                inputs[i, times[1] : times[4], 0] = 1
                theta_noise = angle_noise.rvs(size=(2, times[3] - times[2]))
                inputs[i, times[2] : times[3], 2] = A * np.cos(theta + theta_noise[0])
                inputs[i, times[2] : times[3], 3] = A * np.sin(theta + theta_noise[1])
                inputs[i, times[4] : times[5], 1] = 1
                theta_noise = angle_noise.rvs(size=(2, times[5] - times[4]))
                targets[i, times[4] : times[5], 0] = np.cos(theta + theta_noise[0])
                targets[i, times[4] : times[5], 1] = np.sin(theta + theta_noise[1])
                for j in range(5):
                    periods[i, times[j] : times[j + 1]] = j

                t = times[5]

            inputs[i, :, 0] += flag_noise.rvs(size=(session_length,))
            inputs[i, :, 1] += flag_noise.rvs(size=(session_length,))

        return inputs, targets, periods



class DriscollPlots:

    def memorypro(
        ax, x, y, periods, xcolors, ycolors, single_ax=False, session=0, legend=False, flags=True,
    ):
        """
        Plot stimulus and target from the memory pro task.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or array of Axes
            The axes to plot on.
        x : np.ndarray, shape (n_sessions, session_length, 4), optional
            The input stimuli.
        y : np.ndarray, shape (n_sessions, session_length, 4), optional
            The target stimuli.
        periods : np.ndarray, optional
            The period index for each timepoint.
        xcolors : list
            The colors for the input stimuli.
        ycolors : list
            The colors for the target stimuli.
        single_ax : bool, optional
            Whether to plot all dimensions on the same axis. Default is False.
            If False, then ax should be a one dimensional array of length 3.
        session : int, optional
            The session index to plot. Default is 0.
        legend : bool, optional
            Whether to include a legend. Default is False.
        """
        if x is not None and th.is_tensor(x):
            x = x.cpu().numpy()
        if y is not None and th.is_tensor(y):
            y = y.cpu().numpy()

        if single_ax:
            ax = np.array([ax, ax, ax])

        if x is not None:
            if flags:
                ax[0].plot(x[session, :, 0], color=xcolors[0], label="fixation")
                ax[0].plot(x[session, :, 1], color=xcolors[1], label="response")
            ax[1].plot(x[session, :, 2], color=xcolors[2], label="stim. x")
            ax[1].plot(x[session, :, 3], color=xcolors[3], label="stim. y")
        if y is not None:
            ax[2].plot(y[session, :, 0], color=ycolors[0], label="tgt. x")
            ax[2].plot(y[session, :, 1], color=ycolors[1], label="tgt. y")
        if legend:
            for a in ax:
                vu.legend(a)
        if periods is not None:
            for t in np.where(np.diff(periods[session]) != 0)[0]:
                for a in ax:
                    a.axvline(
                        t,
                        color=".6" if periods[session, t] == 0 else ".9",
                        lw=1,
                        zorder=-1,
                    )

    memorypro_xcolors = [getc("k"), getc("grey"), getc("tab20b:17"), getc("tab20b:19")]
    memorypro_ycolors = [getc("tab20c:0"), getc("tab20c:2")]

def periwindows(signal, flag, radius):
    """
    Extract windows of signal centered at ixs

    Parameters
    ----------
    signal : np.ndarray, shape (..., n_samples, n_features)
        The signal to extract windows from.
    flag : np.ndarray of bool, shape (..., n_samples)
        A binary flag where 1 indicates the center of a window.
    radius : int
        The radius of the windows to extract.
    pad : float, optional
        The value to pad the windows with. Default is 0.

    Returns
    -------
    windows : np.ndarray, shape (n_windows, 2*radius + 1, n_features)
        The extracted windows.
    """
    # pad signal along time axis
    pad = np.zeros(signal.shape[:-2] + (radius, signal.shape[-1]))
    signal = np.concatenate([pad, signal, pad], axis=-2)

    # create windowed signal and selected flagged timepoints
    # shape (..., n_samples, n_features, 2*radius + 1)
    windows = np.lib.stride_tricks.sliding_window_view(signal, 2 * radius + 1, axis=-2)
    # shape (n_windows, n_features, 2*radius + 1)
    windows = windows[flag].transpose(0, 2, 1)

    return windows


def period_start_mask(periods, period):
    """
    Create a mask that is on at the first step during which the period is
    active.

    Parameters
    ----------
    periods : np.ndarray, shape (..., n_samples)
        The period index for each sample.
    period : int
        The period index to create the mask for.

    Returns
    -------
    mask : np.ndarray, shape (..., n_samples)
        The mask for the start of the period.
    """
    pad = np.zeros(periods.shape[:-1], dtype=bool)[..., None]
    change = np.diff((periods == period).astype("int"), axis=-1) == 1
    return np.concatenate([pad, change], axis=-1)


def periperiod_sliced(data, periods):
    """
    Create slices extending from the beginning of the previous period to the end
    of the current period.

    Parameters
    ----------
    data : np.ndarray, shape (..., n_samples, n_features)
        The data to slice.
    periods : np.ndarray, shape (..., n_samples)
        The period index for each sample.

    Returns
    -------
    slices : dict of list of slice
        For each period index, a list of peri-start slices.
    """
    slices = {x: [] for x in np.unique(periods)}
    counter = {x: 0 for x in np.unique(periods)}
    for ix in np.ndindex(periods.shape[:-1]):
        # first timepoint of each period
        starts = np.where(np.diff(periods[ix]) != 0)[0] + 1
        starts = np.concatenate([[0], starts, [periods.shape[-1]]])
        for i in range(1, len(starts) - 1):
            p_ix = periods[ix][starts[i]]
            data_window = data[ix][starts[i - 1] : starts[i + 1]]
            data_t = np.arange(starts[i - 1], starts[i + 1]) - starts[i]
            slices[p_ix].append(
                pd.DataFrame(data_window).assign(rel_time=data_t, number=counter[p_ix])
            )
            counter[p_ix] = counter[p_ix] + 1
    slices = {k: pd.concat(v) for k, v in slices.items()}
    return slices


from line_profiler import LineProfiler

profiler = LineProfiler()


@profiler
def split_trials(data, periods, start_period=0, window=False):
    """
    Create slices of data indexed by trial and period.

    Parameters
    ----------
    data : np.ndarray, shape (..., n_samples, n_features) or dict of arrays
        The data to slice. If a dictionary, may not contain the key 'period'.
    periods : np.ndarray, shape (..., n_samples)
        The period index for each sample.
    start_period : int, optional
        The period index indicating the start of a trial. Default is 0.
    return_sequence : False or tuple of ints
        If truthy, return instead of each period a window of periods
        with radius `return_sequence`.

    Returns
    -------
    trials : list of dict
        A list of trials, each containing 'data' and 'period' keys which map to
        lists of data slices and period indices for each period in each trial.
        If data is a dictionary, then instead of a 'data' key this will contain
        the keys of the given dictionary mapping to lists of corresponding
        sliced arrays. Trials also have key 'source_ix' which is the session index
        from whence the trial came.

    """
    if not isinstance(data, dict):
        data = {"data": data}

    # if `window` is falsy do not return windows
    get_window = window is not False
    window = window if get_window else 0

    trials = []
    for ix in np.ndindex(periods.shape[:-1]):
        # get first timepoint of each period
        starts = np.where(np.diff(periods[ix]) != 0)[0] + 1
        starts = np.concatenate([[0], starts, [periods.shape[-1]]])
        # initialize trial
        trials.append({"period": [], "source_ix": ix, **{k: [] for k in data}})

        # iterate over periods by start timepoint
        for i in range(window, len(starts) - window - 1):
            p_ix = periods[ix][starts[i]]
            # possibly initialize trial
            if p_ix == start_period and i != window:
                trials.append({"period": [], "source_ix": ix, **{k: [] for k in data}})
            # insert data into trial
            if len(trials) == 0:  # no trial has started yet
                continue

            # record data for each period in the window
            for k in data:
                if get_window:
                    data_window = [
                        data[k][ix][starts[i + j] : starts[i + j + 1]]
                        for j in range(-window, window + 1)
                    ]
                else:
                    data_window = data[k][ix][starts[i] : starts[i + 1]]
                trials[-1][k].append(data_window)
            # record type of period for each period in the window
            if get_window:
                p_window = [
                    periods[ix][starts[i + j]] for j in range(-window, window + 1)
                ]
            else:
                p_window = p_ix
            trials[-1]["period"].append(p_window)

    return trials


def extract_trial_data(
    f,
    trials,
    data_key="stim",
    window=None,
    period_i=2,
    cmap=None,
    color_range=(0, 2 * np.pi),
    n_clusters=None,
):
    """

    Parameters:
    f : function
        A function to apply to each trial / period returning a scalar.
    trials : list of dict
        As returned by `split_trials`
    data_key : str
        The key in the trial dictionary to extract the data from
    window : int or None
        The window of data present in `trials`, as passed to `split_trials`.
    period_i : int
        The index of the period in the period array to extract the data from.
        Note this is different than the type of period to select.
    cmap : str or None
        The colormap to use for coloring the data. If none, no colors array will
        be returned.
    color_range : tuple of float
        The range of values to normalize the colormap to.
    n_clusters : int
        The number of clusters to sort the data into. If None, do return a
        partition of the trials.

    """
    if window is not None:
        trial_angles = np.array([f(t[data_key][period_i][window]) for t in trials])
    else:
        trial_angles = np.array([f(t[data_key][period_i]) for t in trials])
    ret = (trial_angles,)

    if cmap is not None:
        ret = ret + (
            cmap((trial_angles - color_range[0]) / (color_range[1] - color_range[0])),
        )
    if n_clusters is not None:
        clusts = np.array_split(np.argsort(trial_angles), n_clusters)
        ret = ret + (clusts,)
    return ret


def nanstack(x, axis=0):
    """
    Parameters:
    x : list of np.ndarray
        A list of arrays to stack. Each array must have the same shape.
    axis : int
        The axis to stack the arrays along.
    """
    if axis < 0:
        axis = x[0].ndim + axis
    # x is an array of scalars
    if np.array(x[0]).ndim == 0:
        return np.array(x)
    x = [np.array(t) for t in x]

    max_len = max(t.shape[axis] for t in x)
    x = [
        np.concatenate(
            [
                y,
                np.full(
                    y.shape[:axis] + (max_len - y.shape[axis],) + y.shape[axis + 1 :],
                    np.nan,
                ),
            ],
            axis=axis,
        )
        for y in x
    ]
    return np.array(x)


def apply_to_trial_groups(f, trials, groups, as_array=False):
    """
    Apply a function to each group of trials.

    Only accepts windowed data.

    Parameters:
    f : function
        A function to apply to each group of trials. To match structure of
        `trials`, this should return a dictionary of arrays or lists of arrays.
    trials : list of dict
        As returned by `split_trials`.
    groups : list of list of int
        A list of groups of trial indices to apply the function to.
    as_array: bool
        If True, pass data to f as an array of shape (n_trials, n_time, ...)
        with n_time padded for each trial and phase with nans.

    """
    ret = []
    window = len(trials[0]["period"][0])

    for group in groups:

        # check each has same period structure
        n_period = len(trials[group[0]]["period"])
        assert all(
            len(trials[i]["period"]) == n_period for i in group
        ), "All trials in group must have the same period structure"

        if as_array:
            stackfn = nanstack
        else:
            stackfn = lambda x: x

        # trials[i_trial]["data_key"][i_period][i_window][time, ...]
        # data["data_key"][i_period][i_window][i_trial (in group), time, ...]
        # or if not as_array
        # data["data_key"][i_period][i_window][i_trial (in group)][time, ...]
        # then apply f:
        # data["data_key"][i_period][i_window][...f_return_shape...]
        data = {k: [] for k in trials[0]}
        for k in data:
            # if this is trial metadata (no iteration over periods) then skip it
            if not hasattr(trials[0][k], "__getitem__") or not hasattr(
                trials[0][k][0], "__getitem__"
            ):
                continue
            for j in range(n_period):
                data[k].append([])
                for l in range(window):
                    data[k][j].append(f(stackfn([trials[i][k][j][l] for i in group])))

        ret.append(data)
    return ret


def nanmean_with_req(x, n_req=0, **kws):
    """
    kws: any kewords passable to both nanmean and count_nonzero
    """
    ret = np.nanmean(x, **kws)
    if n_req > 0 and ret.ndim:
        invalid = (
            np.count_nonzero(
                np.isfinite(x),
                **kws,
            )
            <= n_req
        )
        ret[invalid] = np.nan
    return ret
