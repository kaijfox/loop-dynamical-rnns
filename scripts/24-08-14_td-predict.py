"""

Train a discounted sum network on a DriscollMultiBlockActivity dataset.

Usage:
    <root_dir> <dset_hash> <output_fmt> <gamma> <network_type> <network_args>
    <train_args>
    
Arguments:
root_dir: str
    Root directory for the repository.
dset_hash: str
    Hash of the dataset to use.
output_fmt: str
    Format string for the output path, relative to the root directory. Should
    contain a `{hash}` substring to insert the hash of the saved model.
gamma: float
    Discount factor for the discounted sum.
network_type: str
    Type of network to use, as supported by
    `dynrn.predictors.create_predictor_network`.
network_args: dict
    Arguments to pass to the network constructor
    `dynrn.predictors.create_predictor_network` that can be evaluated as
    `dict(<train_args>)`.
train_args: dict
    Arguments to pass to the training function, that can be evaluated as
    `dict(<train_args>)`.
"""

import torch as th
import torch.jit as jit
import numpy as np
from torch import optim
from torch import nn
from dynrn.rnntasks import DriscollTasks, itiexp
from dynrn.predictors import create_predictor_network, MultiBlockActivityDataset, fit_dsn, save_dsn
import dynrn.basic_rnns as rnns
import scipy.stats
import dill
from scipy.stats import uniform, norm
from datetime import datetime
from mplutil import util as vu
import matplotlib.pyplot as plt
import scipy.stats
from pathlib import Path
import tqdm
import joblib as jl
import time
import seaborn as sns
from dynrn.basic_rnns import timehash, hash_or_path, find_hash
import sys


# cuda setup
device = th.device("cuda" if th.cuda.is_available() else "cpu")
cpu = th.device("cpu" if th.cuda.is_available() else "cpu")
print("Using device:", device.type)

# -------- Process args

root_dir = Path(sys.argv[1])
act_hash = sys.argv[2]
dsn_path_fmt = sys.argv[3]
gamma = float(sys.argv[4])
network_type = sys.argv[5]
network_args = eval(f"dict({sys.argv[6]})")
train_args = {
    **dict(
        lr=1e-3,
        steps=601,
        checkpt=100,
        n=1,
    ),
    **eval(f"dict({sys.argv[7]})"),
}

# -------- Load source data and network

act_path = find_hash(root_dir, act_hash, '.dil')
act_dataset: MultiBlockActivityDataset = dill.load(open(act_path, "rb"))
act_data = act_dataset['train']

create_model = create_predictor_network(network_type, network_args, act_data['n_act'])


# -------- Training setup

for i_net in range(train_args["n"]):
    
    th.manual_seed(i_net)
    predictor = jit.script(create_model())

    cumulant_fn = lambda acts: acts[:, 1:]
    x = th.tensor(act_data['activity'], dtype=th.float32)
    y = cumulant_fn(x)

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
    dsn_hash = timehash(unique_within=root_dir, ext='.pt')
    dsn_path = root_dir / dsn_path_fmt.format(hash=dsn_hash)
    save_dsn(
        dsn_path,
        predictor,
        dataset=None,
        gamma=gamma,
        cumulant_fn=cumulant_fn,
        checkpoints=_ckpts,
        losses=_losses,
        source_meta = {
            "act_hash": act_hash,
            "rnn_hash": act_dataset["rnn_hash"],
            "dataset_hash": act_dataset["dataset_hash"],
            "train_args": train_args,
            "network_args": network_args,
            "repo_status": rnns.gitinfo(__file__),
        }
    )
    print(f"Saved DSN model: {dsn_path}")

