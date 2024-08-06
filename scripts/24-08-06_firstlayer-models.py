"""
Train discounted sum predictors with input bottleneck on an activity dataset.

usage <dataset> <output_root> <output_path> <bottleneck> <gamma>
    <network_type> <network_args> <train_args>

Parameters
----------
dataset:
    Can be provided in form <root>:<hash> to search for with a hash or can be a
    full path.
output_root:
    Root directory to save the model, within which the hash will be unique.
output_path:
    Path to save the model within the root, can contain a substring `{hash}` to
    insert a time-based hash.
bottleneck, int:
    Number of units in the bottleneck layer.
gamma, float:
    Discount factor for the discounted sum.
network_type, str:
    Type of network to use, currently only `softplus` is supported.
network_args, dict:
    Arguments to pass to the network constructor.
train_args, dict:
    Arguments to pass to the training function.

Network types and corresponding arguments:
- `softplus`
    - `widths`: list of two integers, the widths of the two hidden layers
Train args:
- `lr`: float, default 1e-3, learning rate
- `steps`: int, default 601, number of training steps
- `checkpt`: int, default 100, checkpoint every n steps
    

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
from dynrn.basic_rnns import timehash, hash_or_path
import scipy.stats
from scipy.stats import uniform, norm
from datetime import datetime
from mplutil import util as vu
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import tqdm
import os
import joblib as jl
import time
import seaborn as sns
import sys

# cuda setup
device = th.device("cuda" if th.cuda.is_available() else "cpu")
cpu = th.device("cpu" if th.cuda.is_available() else "cpu")
print("Using device:", device.type)


# -------- Process args
dset_path, dset_hash = hash_or_path(sys.argv[1])
dsn_root = Path(sys.argv[2])
dsn_path_fmt = sys.argv[3]
bottleneck = int(sys.argv[4])
gamma = float(sys.argv[5])
network_type = sys.argv[6]
network_args = eval(f"dict({sys.argv[7]})")
train_args = {
    **dict(
        lr=1e-3,
        steps=601,
        checkpt=100,
    ),
    **eval(f"dict({sys.argv[8]})"),
}

# -------- Load source data
data = jl.load(dset_path)["dataset"]


# -------- Define model
if network_type == "softplus":
    widths = network_args.get("widths", None)
    if len(widths) != 2:
        raise ValueError(
            f"`widths` network argument must" " be a list of length 2, got {widths}"
        )
    layers = lambda: (
        nn.Linear(data.train.n_act, bottleneck),
        nn.Linear(bottleneck, widths[0]),
        nn.Softplus(),
        nn.LayerNorm(widths[0]),
        nn.Linear(widths[0], widths[1]),
        nn.Softplus(),
        nn.Linear(widths[1], data.train.n_act),
    )
    th.manual_seed(0)
    predictor = jit.script(nn.Sequential(*layers()))


# -------- Training setup
cumulant_fn = lambda d: d.activity[:, 1:]

x = data.train.activity
y = cumulant_fn(data.train)

predictor.to(device)
opt = optim.Adam(predictor.parameters(), lr=train_args["lr"])

# -------- Train
_losses, _ckpts = fit_dsn(
    predictor,
    x.to(device),
    y.to(device),
    opt,
    gamma=gamma,
    n_steps=train_args["steps"],
    loss_fn=nn.MSELoss(),
    checkpoint_every=train_args["checkpt"],
)


# -------- Save
dsn_hash = timehash(unique_within=dsn_root)
dsn_path = dsn_root / dsn_path_fmt.format(hash=dsn_hash)
save_dsn(
    dsn_path,
    predictor,
    data,
    gamma=gamma,
    cumulant_fn=cumulant_fn,
    checkpoints=_ckpts,
    losses=_losses,
)
print(f"Saved DSN model: {dsn_path}")