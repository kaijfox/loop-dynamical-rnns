import torch as th
import torch.jit as jit
import copy
import numpy as np
import dill
from collections import namedtuple
from torch import optim
from torch import nn
from dynrn.rnntasks import DriscollTasks, itiexp
from dynrn.predictors import (
    activity_dataset,
    save_dsn,
    load_dsn,
    create_memorypro_activity_dataset,
    fit_dsn,
    td_loss,
)
import dynrn.basic_rnns as rnns
from dynrn.basic_rnns import timehash, find_hash, hash_or_path
from dynrn.viz import styles
import scipy.stats
from scipy.stats import uniform, norm
from datetime import datetime
from scipy.spatial.distance import cosine as cosine_dist
from mplutil import util as vu
import scipy.linalg
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from cmap import Colormap
import tqdm
import os
import joblib as jl
import time
import glob
import seaborn as sns
import sys



def plot_loss(test_losses, traindata, colors=None):
    if colors is None:
        colors = styles.default(colors)
    
    steps = np.array(sorted(list(test_losses.keys())))
    _losses = {i: test_losses[i].mean() for i in steps}

    fig, ax = plt.subplots(1, 1, figsize=(2, 1.5))
    rnns.plot_rnn_training(
        _losses,
        None,
        np.empty([0, 0, 0]),
        epochs=steps,
        ax=[ax],
        loss_matches_epochs=True,
    )
    ax.set_yscale("log")
    # two possible formats for saved losses (1D, or 2D with epoch included)
    if np.array(traindata["losses"]).ndim > 1:
        ax.plot(*traindata["losses"][::-1], color=colors.neutral, lw=0.3, zorder=-1)
    else:
        ax.plot(traindata["losses"], color=colors.neutral, lw=0.3, zorder=-1)
    return fig


def plot_blockwise_loss(test_data, test_losses):
    n_blocks = len(test_data["block_slices"])
    task_colors = Colormap("crest")(np.linspace(0.2, 0.8, n_blocks))
    steps = np.array(sorted(list(test_losses.keys())))

    fig, ax = plt.subplots(1, 1, figsize=(3, 1.6))
    for i in range(n_blocks):
        block_losses = {
            j: test_losses[j][test_data["block_slices"][i]].mean() for j in steps
        }
        task = test_data["block_tasks"][i]
        _lbl = (
            f"{i} {task.short_name if hasattr(task, 'short_name') else task.__name__}"
        )
        ax.plot(
            steps, [block_losses[i] for i in steps], color=task_colors[i], label=_lbl
        )
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Block loss")
    leg = vu.legend(ax, fontsize=6, title="block")
    plt.setp(leg.get_title(), fontsize=7)
    return fig


def plot_block_examples(test_data, test_losses, test_preds, session=0):
    n_blocks = len(test_data["block_slices"])
    block_data = DriscollTasks.split_dataset(test_data)
    max_nax = max(
        len(_task.stim_groups) + 2 * len(_task.tgt_groups)
        for _task in [block_data[i]["task"] for i in range(n_blocks)]
    )
    bigfig, figs, _ = vu.flat_subfig_grid(n_blocks, 5, (4, max_nax * 0.75))
    for ibl in range(n_blocks):
        fig = figs[ibl]
        task: DriscollTasks.DriscollTask = block_data[ibl]["task"]
        steps = np.array(sorted(list(test_losses.keys())))
        best_step = sorted(steps, key=lambda i: test_losses[i].mean())[0]
        nax = len(task.stim_groups) + 2 * len(task.tgt_groups)
        _, ax, _ = vu.flat_grid(nax, 1, ax_size=None, fig=fig, sharex=True)

        for a1, a2 in zip(ax[-len(task.tgt_groups) :], ax[len(task.stim_groups) :]):
            a2.sharey(a1)
            a1.set_ylabel("rnn outputs")
            a2.set_ylabel("targets")
        DriscollTasks.plot_session(block_data[ibl], session=session, ax=ax, legend=True)
        _pred = test_preds[best_step][ibl][[session]]
        # arrange axes for plot_session to only plot outputs (which it thinks are targets)
        _predax = [None] * len(task.stim_groups) + ax[-len(task.tgt_groups) :].tolist()
        DriscollTasks.plot_session(task, y=_pred, session=0, ax=_predax, legend=False)
        _lbl = (
            f"{ibl} {task.short_name if hasattr(task, 'short_name') else task.__name__}"
        )
        ax[0].set_title(_lbl, fontsize=8)
    return bigfig



def lowrank_space(rnn: rnns.BasicRNN_LR):
    """
    Projections to and from the input span of a low-rank RNN.

    The low-rank (i.e. decoder) space is the the null-space complement of $W$.
    When $W = UV^T$, this can be parameterized by the pseudoinverse of $V$, but
    it can be more convenient to to parameterize it by the SVD $W = Q \Sigma
    P^T$, which can be thought of as defining an equivalent low rank network
    that only varies original in its encoder / decoder spaces.

    Parameters
    ----------
    rnn : rnns.BasicRNN_LR
        Low-rank RNN, with low rank decoder weights rnn.h2h.v, having shape
        (n_hidden, rank)
    
    Returns
    -------
    down : np.ndarray, shape (rank, n_hidden)
        Projection from hidden activations to equivalent orthogonal decoder
        space of the low-rank RNN
    up : np.ndarray, shape (n_hidden, rank)
        Projection from equivalent orthogonal decoder space to hidden
        activations
    encoder : np.ndarray, shape (n_hidden, rank)
        Encoder weights for the equivalent orthogonal low rank network.
    """
    # SVD of the decoder weights
    Q, s, Pt = th.linalg.svd(rnn.h2h.u @ rnn.h2h.v.T,)
    P = Pt.T
    Q = Q[:, :rnn.rank]
    s = s[:rnn.rank]
    P = P[:, :rnn.rank]
    # the orthogonal encoder weights
    encoder = Q @ th.diag(s)
    # the projection to the low-rank space
    down = P.T
    # the projection from the low-rank space
    up = P
    return dict(
        down = down.detach().cpu().numpy(),
        up = up.detach().cpu().numpy(),
        encoder = encoder.detach().cpu().numpy(),
    )

def _space_and_x_tensors(rnn, space, x, n_samples):
    """Convert args to tensor if necessary."""

    if space is None:
        space = lowrank_space(rnn)
    if not th.is_tensor(space["down"]):
        down = th.tensor(space["down"], dtype=th.float32)
        up = th.tensor(space["up"], dtype=th.float32)
        enc = th.tensor(space["encoder"], dtype=th.float32)
    else:
        down = space["down"]
        up = space["up"]
        enc = space["encoder"]

    if isinstance(n_samples, int):
        n_samples = (n_samples,)
    if x is None:
        x = th.zeros(n_samples + (rnn.nx,), dtype=th.float32)
    elif not th.is_tensor(x):
        x = th.tensor(x, dtype=th.float32)
    
    return down, up, enc, x

def lowrank_step(h_low, rnn: rnns.BasicRNN_LR = None, space: dict = None, x = None):
    """
    Calculate motion within decoder space of a low-rank RNN.

    Parameters
    ----------
    h_low : np.ndarray or th.tensor, shape (n_samples, rank)
        Coordinates in decoder space, i.e. V^T output space.
    rnn : rnns.BasicRNN_LR, optional
        Low-rank RNN. Required only if `space` or `x` are not provided.
    space : dict, optional
        Dictionary containing the encoder, up, and down matrices for the low-rank
        RNN as returned by `lowrank_space`. If not provided, these are
        calculated from `rnn`.
    x : np.ndarray or th.tensor, optional
        Input to the RNN. If not provided, defaults to zeros.
    """
    

    if not th.is_tensor(h_low):
        h_low = th.tensor(h_low, dtype=th.float32)
    down, up, _, x = _space_and_x_tensors(rnn, space, x, h_low.shape[:-1])

    # --- compute next step activations
    h = h_low @ up.T # (n_samples, n_hidden)
    _, h_new = rnn.forward(x, h) # (n_samples, n_hidden)
    h_low_new = h_new @ down.T # (n_samples, rank)

    return h_low_new.detach().cpu().numpy()


def lowrank_trajectories(rnn, n_traj, n_steps, h_rng, space = None, x = None, seed: int = 0):

    rng = np.random.default_rng(seed)
    down, up, _, x = _space_and_x_tensors(rnn, space, x, n_traj)
    x = th.tile(x[:, None, :], (1, n_steps, 1))
    h0_low = rng.uniform(*h_rng, size=(n_traj, rnn.rank))
    h0_low = th.tensor(h0_low, dtype=th.float32)
    h0 = h0_low @ up.T
    
    _, trajs = rnn.seq_forward(x, h0) # shape (n_traj, n_steps, n_hidden)
    h_low = trajs @ down.T # shape (n_traj, n_steps, rank)
    h_low = th.concat([h0_low[:, None, :], h_low], dim=1)

    return h_low.detach().cpu().numpy()




