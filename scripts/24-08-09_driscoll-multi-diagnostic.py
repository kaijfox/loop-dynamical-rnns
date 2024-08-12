"""
Diagnostic plots for block-style driscoll tasks from `driscoll-multi-rnn.py`.

Usage:
    <root_dir> <rnn_path> <output_dir>
"""

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

# plotting setup
from dynrn.viz import styles
from dynrn.viz.styles import getc

colors, plotter = styles.init_plt(".", fmt="pdf", display=False)
plot_root = Path(plotter.plot_dir)

# cuda setup
device = th.device("cuda" if th.cuda.is_available() else "cpu")
cpu = th.device("cpu" if th.cuda.is_available() else "cpu")
print(device.type)


# -------- Process args

root_dir = Path(sys.argv[1])
rnn_hash = sys.argv[2]
output_dir = Path(sys.argv[3])


# -------- Load the rnn and source data

rnn_path = find_hash(root_dir, rnn_hash, ".pt")
rnn_path = Path(str(rnn_path)[:-3])  # remove .pt
final_rnn, rnn_ckpts, traindata = rnns.load_rnn(rnn_path, device=device)

task_data_hash = traindata["dataset_hash"]
task_data = dill.load(open(find_hash(root_dir, task_data_hash, ".dil"), "rb"))
task_meta_hash = traindata["task_hash"]
task_meta = dill.load(open(find_hash(root_dir, task_meta_hash, ".dil"), "rb"))


# -------- Evalutate loss / predictions on test data

test_data: DriscollTasks.MultiTaskBlockDataset = task_data["test"]
n_ex_session = 1

test_preds = {}
test_losses = {}

x = th.tensor(test_data["stimuli"], dtype=th.float32).to(device)
h_init = th.zeros([x.shape[0], final_rnn.nh])
for step, rnn in tqdm.tqdm(rnn_ckpts.items()):
    all_preds, all_h = rnn.seq_forward(x, h_init)
    test_losses[step] = (all_preds.detach() - test_data["targets"]) ** 2
    test_preds[step] = [
        all_h[slc][:n_ex_session].detach().numpy() for slc in test_data["block_slices"]
    ]


# -------- Plotting

def plot_loss():
    block = 0
    session = 0
    n_exsteps = 3

    # _tgt = test_data['targets'][test_data['block_slices'][block]][[session]]
    steps = np.array(sorted(list(test_losses.keys())))
    # _yhats = {i: test_preds[i][block][[session]] for i in _steps}
    _losses = {i: test_losses[i].mean() for i in steps}

    fig, ax = plt.subplots(1, 1, figsize = (2, 1.5))
    rnns.plot_rnn_training(
        _losses,
        None,
        np.empty([0, 0, 0]),
        epochs = steps,
        ax = [ax],
        loss_matches_epochs=True
    )
    ax.set_yscale('log')
    ax.plot(traindata['losses'], color = colors.neutral, lw = 0.3, zorder = -1)
    return fig
    

def plot_blockwise_loss():
    n_blocks = len(test_data['block_slices'])
    task_colors = Colormap('crest')(np.linspace(0.2, 0.8, n_blocks))
    steps = np.array(sorted(list(test_losses.keys())))

    fig, ax = plt.subplots(1,1, figsize=(3,1.6))
    for i in range(n_blocks):
        block_losses = {i: test_losses[i][test_data['block_slices'][i]].mean() for i in steps}
        task = test_data['block_tasks'][i]
        _lbl = f"{i} {task.short_name if hasattr(task, 'short_name') else task.__name__}"
        ax.plot(steps, [block_losses[i] for i in steps], color = task_colors[i], label = _lbl)
    ax.set_yscale('log')
    ax.set_xlabel("Training step")
    ax.set_ylabel("Block loss")
    leg = vu.legend(ax, fontsize = 6, title='block')
    plt.setp(leg.get_title(), fontsize=7)
    return fig
    
def plot_block_examples(session = 0):
    n_blocks = len(test_data['block_slices'])
    block_data = DriscollTasks.split_dataset(test_data)
    max_nax = max(len(_task.stim_groups) + 2 * len(_task.tgt_groups) for _task in [block_data[i]["task"] for i in range(n_blocks)])
    bigfig, figs, _ = vu.flat_subfig_grid(n_blocks, 5, (4, max_nax * 0.75))
    for ibl in range(n_blocks):
        fig = figs[ibl]
        task: DriscollTasks.DriscollTask = block_data[ibl]["task"]
        steps = np.array(sorted(list(test_losses.keys())))
        best_step = sorted(steps, key=lambda i: test_losses[i].mean())[0]
        nax = len(task.stim_groups) + 2 * len(task.tgt_groups)
        _, ax, _ = vu.flat_grid(nax, 1, ax_size = None, fig = fig, sharex=True)

        for a1, a2 in zip(
            ax[-len(task.tgt_groups) :], ax[len(task.stim_groups) :]
        ):
            a2.sharey(a1)
            a1.set_ylabel("rnn outputs")
            a2.set_ylabel("targets")
        DriscollTasks.plot_session(
            block_data[ibl], session=session, ax=ax, legend=True
        )
        _pred = test_preds[best_step][ibl][[session]]
        # arrange axes for plot_session to only plot outputs (which it thinks are targets)
        _predax = [None] * len(task.stim_groups) + ax[
            -len(task.tgt_groups) :
        ].tolist()
        DriscollTasks.plot_session(task, y=_pred, session=0, ax=_predax, legend=False)
        _lbl = f"{ibl} {task.short_name if hasattr(task, 'short_name') else task.__name__}"
        ax[0].set_title(_lbl, fontsize=8)
    return bigfig