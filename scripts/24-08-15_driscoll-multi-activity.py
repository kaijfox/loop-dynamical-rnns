"""
Diagnostic plots for block-style driscoll tasks from `driscoll-multi-rnn.py`.

Usage:
    <root_dir> <rnn_hash> <dset_hash> <n_dims> <output_fmt>

Args:
root_dir: str
    path to the root directory containing the rnn and task data in which to
    seach hashes.
rnn_hash: str
    hash of the rnn to load.
dset_hash: str
    hash of the task dataset to load.
n_dims: int or str
    number of dimensions to reduce the activity to. If 'pcs' PCA will be
    applied but at full dimensionality. If 'units' no reduction will be applied.
output_fmt: str
    path to save the model within the root, can contain a substring `{hash}` to
    insert a time-based hash.
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
rnn_hash = sys.argv[2]
dset_hash = sys.argv[3]
n_dims = sys.argv[4]
output_fmt = Path(sys.argv[5])

if n_dims == "pcs":
    n_dims = -1
    apply_pca = True
elif n_dims == "units":
    n_dims = -1
    apply_pca = False
else:
    n_dims = int(n_dims)
    apply_pca = True


# -------- Load the rnn and task dataset

rnn_path = find_hash(root_dir, rnn_hash, ".pt")
rnn_path = Path(str(rnn_path)[:-3])  # remove .pt
final_rnn, rnn_ckpts, traindata = rnns.load_rnn(rnn_path, device=device)

dset_path = find_hash(root_dir, dset_hash, ".dil")
task_dataset = dill.load(open(dset_path, "rb"))


# ------ Compute RNN responses

rnn_hash = timehash(unique_within=root_dir, ext=".pt")
rnn_path = root_dir / output_fmt.format(hash=rnn_hash)

activity_datasets = evaluate_multiblock_activity(
    final_rnn,
    n_dim=n_dims,
    apply_pca=apply_pca,
    pca_set="train",
    **task_dataset,
)

# ------ Save out simulated trajectories

act_hash = timehash(unique_within=root_dir, ext=".pt")
act_path = root_dir / output_fmt.format(hash=act_hash)

dill.dump(
    {
        **activity_datasets,
        "source": "driscoll-multi-rnn.py",
        "rnn_hash": rnn_hash,
        "dataset_hash": dset_hash,
        "repo_status": rnns.gitinfo(__file__),
    },
    open(dset_path, "wb"),
)
print("Saved activity dataset to:")
print(dset_path)
