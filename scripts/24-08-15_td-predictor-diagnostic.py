

"""
Diagnostic plots for block-style driscoll tasks from `td-predict.py`.

Usage:
    <root_dir> <dsn_hash> <output_dir>
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
    MultiBlockActivityDataset,
    plot_block_error_examples
)
import dynrn.basic_rnns as rnns
from dynrn.basic_rnns import timehash, find_hash, hash_or_path
from dynrn.driscoll_rnns import plot_loss, plot_blockwise_loss
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
dsn_hash = sys.argv[2]
output_dir = Path(sys.argv[3])


# -------- Load the rnn and source data

dsn_path = find_hash(root_dir, dsn_hash, ".pt")
dsn_path = Path(str(dsn_path)[:-3])  # remove .pt
final_rnn, dsn_ckpts, traindata = rnns.load_rnn(dsn_path, device=device)

task_data_hash = traindata["act_hash"]
task_data = dill.load(open(find_hash(root_dir, task_data_hash, ".dil"), "rb"))

test_data: MultiBlockActivityDataset = task_data["test"]
cumulant_fn = traindata['cumulant_fn']
gamma = traindata['gamma']
n_ex_session = 1

# -------- Evalutate loss / predictions on test data

test_preds = {}
test_losses = {}

x = th.tensor(test_data["activity"], dtype=th.float32).to(device)
cumul_gt = cumulant_fn(test_data["activity"])
for step, predictor in tqdm.tqdm(dsn_ckpts.items()):
    # calculate cumulants from predicted discounted sums
    sum_preds = predictor(x).cpu().detach().numpy()
    cumul_pr = sum_preds[:, :-1] - gamma * sum_preds[:, 1:]

    loss_series = (cumul_gt - cumul_pr) ** 2
    test_losses[step] = loss_series
    test_preds[step] = cumul_pr

# ----- Generate and save plots

if not output_dir.exists():
    output_dir.mkdir(parents=True)
    print(f"Warning: created output directory {output_dir}")
net_filename = dsn_path.name
if net_filename.endswith(".pt"):
    net_filename = net_filename[:-3]
output_dir = output_dir / net_filename
output_dir.mkdir(parents=True, exist_ok=True)

print("test_losses", test_losses.keys())
finalize_kw = dict(path=output_dir, transparent=True)
plotter.finalize(plot_loss(test_losses, traindata), f"loss_{dsn_hash}", **finalize_kw)
plotter.finalize(plot_blockwise_loss(test_data, test_losses), f"blockloss_{dsn_hash}", **finalize_kw)
plotter.finalize(plot_block_error_examples(test_data, cumul_gt, test_losses, test_preds), f"ex_{dsn_hash}", **finalize_kw)

