"""
Diagnostic plots for block-style driscoll tasks from `driscoll-multi-rnn.py`.

Usage:
    <root_dir> <task_hash> <n_train> <n_test> <session_length> <base_seed> <output_fmt>

Args:
root_dir: str
    path to the root directory containing the rnn and task data in which to
    seach hashes.
task_hash: str
    hash of the task keyword arguments to load.
n_train, n_test: int
    number of training and testing sessions to simulate.
session_length: int
    number of timesteps
base_seed: int
    seed for the random number generator
output_fmt: str
    path to save the model within the root, can contain a substring `{hash}` to
    insert a time-based hash. Ending with ".dil"
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
from dynrn.predictors import evaluate_multiblock_activity
import dynrn.basic_rnns as rnns
from dynrn.basic_rnns import timehash, find_hash, hash_or_path
from dynrn.driscoll_rnns import plot_loss, plot_blockwise_loss, plot_block_examples
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
task_hash = sys.argv[2]
n_train = int(sys.argv[3])
n_test = int(sys.argv[4])
length = int(sys.argv[5])
seed = int(sys.argv[6])
output_fmt = sys.argv[7]


# -------- Load task metadata and sample stimuli / targets

task_kws = dill.load(open(find_hash(root_dir, task_hash, ".dil"), "rb"))


battery = task_kws.keys()
rng = np.random.default_rng(seed=seed)
seeds = rng.integers(0, 2**32 - 1, (2, len(battery)), dtype=np.uint32)

multitask = lambda n_sess, sess_len, seeds: DriscollTasks.merge_datasets(
    [
        DriscollTasks.generate_sessions(t, n_sess, sess_len, task_kws[t], seed=seeds)
        for s, t in zip(seeds, battery)
    ]
)
dataset = {
    "train": multitask(n_train, length, seeds[0]),
    "test": multitask(n_test, length, seeds[1]),
}


# ------ Save out simulated trajectories

dset_hash = timehash(unique_within=root_dir, ext=".dil")
dset_path = root_dir / output_fmt.format(hash=dset_hash)

dill.dump(
    {
        **dataset,
        "task_hash": task_hash,
        "seeds": {'train': seeds[0], 'test': seeds[1]},
    },
    open(dset_path, "wb"),
)
print("Wrote:")
print(dset_path)
