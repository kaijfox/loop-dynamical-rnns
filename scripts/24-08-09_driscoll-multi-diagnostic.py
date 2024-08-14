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
output_dir = Path(sys.argv[3])


# -------- Load the rnn and source data

rnn_path = find_hash(root_dir, rnn_hash, ".pt")
rnn_path = Path(str(rnn_path)[:-3])  # remove .pt
final_rnn, rnn_ckpts, traindata = rnns.load_rnn(rnn_path, device=device)

task_data_hash = traindata["dataset_hash"]
task_data = dill.load(open(find_hash(root_dir, task_data_hash, ".dil"), "rb"))


# -------- Evalutate loss / predictions on test data

test_data: DriscollTasks.MultiTaskBlockDataset = task_data["test"]
n_ex_session = 1

test_preds = {}
test_losses = {}

x = th.tensor(test_data["stimuli"], dtype=th.float32).to(device)
h_init = th.zeros([x.shape[0], final_rnn.nh]).to(device)
for step, rnn in tqdm.tqdm(rnn_ckpts.items()):
    all_preds, all_h = rnn.seq_forward(x, h_init)
    test_losses[step] = (all_preds.cpu().detach() - test_data["targets"]) ** 2
    test_preds[step] = [
        all_preds[slc][:n_ex_session].cpu().detach().numpy() for slc in test_data["block_slices"]
    ]

# ----- Generate and save plots

if not output_dir.exists():
    output_dir.mkdir(parents=True)
    print(f"Warning: created output directory {output_dir}")
net_filename = rnn_path.name
if net_filename.endswith(".pt"):
    net_filename = net_filename[:-3]
output_dir = output_dir / net_filename
output_dir.mkdir(parents=True, exist_ok=True)


finalize_kw = dict(path=output_dir, transparent=True)
plotter.finalize(plot_loss(test_losses, traindata), f"loss_{rnn_hash}", **finalize_kw)
plotter.finalize(plot_blockwise_loss(test_data, test_losses), f"blockloss_{rnn_hash}", **finalize_kw)
plotter.finalize(plot_block_examples(test_data, test_losses, test_preds), f"ex_{rnn_hash}", **finalize_kw)
