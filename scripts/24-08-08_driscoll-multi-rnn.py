"""
Train a recurrent neural network on the Driscoll MultiTaskBlockDataset.

Usage:
    <dset_path> <rnn_root> <rnn_path_fmt> <network_type> <network_args>
    <train_args>
    
Parameters
---------
dset_path: str
    Can be provided in form <root>:<hash> to search for dataset with a hash or
    can be a full path.
rnn_root: str
    path to the root directory to save the model
rnn_path_fmt: str
    path to save the model within the root, can contain a substring `{hash}` to
    insert a time-based hash.
network_type: str
    Type of network to use, currently only `lnrnn` is supported.
network_args: dict
    Arguments to pass to the network constructor, as a string that can be
    evaluated to a dictionary when surrounded by `dict()`.
train_args: dict
    Arguments to pass to the training function, as a string that can be
    evaluated to a dictionary when surrounded by `dict()`.
"""

import torch as th
import torch.jit as jit
import numpy as np
from torch import optim
from torch import nn
from dynrn.rnntasks import DriscollTasks, itiexp
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
from dynrn.basic_rnns import timehash, hash_or_path
import sys


# cuda setup
device = th.device("cuda" if th.cuda.is_available() else "cpu")
cpu = th.device("cpu" if th.cuda.is_available() else "cpu")
print("Using device:", device.type)

# -------- Process args

dset_path, dset_hash = hash_or_path(sys.argv[1], ext=".dil")
rnn_root = Path(sys.argv[2])
rnn_path_fmt = sys.argv[3]
network_type = sys.argv[4]
network_args = eval(f"dict({sys.argv[5]})")
train_args = {
    **dict(
        lr=1e-4,
        steps=601,
        checkpt=100,
        n=1,
        batch=None,
    ),
    **eval(f"dict({sys.argv[6]})"),
}

# -------- Load source data

data = dill.load(open(dset_path, "rb"))
task_hash = data["task_hash"]
data: DriscollTasks.MultiTaskBlockDataset = data["train"]

# -------- Define model

n_stim = data["n_stim"] + data["n_task_flags"]
n_tgt = data["n_tgt"]
if network_type == "lnrnn":
    # Arg defaults
    args = {
        **dict(
            alpha=1,
            act="Softplus",
            bias=True,
            nh=1024,
        ),
        **network_args,
    }
    args["act"] = getattr(nn, args["act"])()
    # Define and initialize
    th.manual_seed(0)

    def init_hidden(batch_size):
        return th.zeros(batch_size, args["nh"], device=device)

    def create_model():
        rnn = rnns.BasicRNN_LN(
            n_stim,
            args["nh"],
            n_tgt,
            alpha=args["alpha"],
            act=args["act"],
            bias=args["bias"],
        )
        with th.no_grad():
            W = scipy.stats.ortho_group.rvs(rnn.nh, random_state=0)
            rnn.h2h.weight.data = th.tensor(W, dtype=th.float32)
        # Jit and transfer
        rnn.to(device)
        return jit.script(rnn)


# -------- Training setup

x = th.tensor(data["stimuli"], dtype=th.float32).to(device)
y = th.tensor(data["targets"], dtype=th.float32).to(device)

for i_net in range(train_args["n"]):

    th.manual_seed(i_net)
    rnn = create_model()
    opt = optim.Adam(rnn.parameters(), weight_decay=0, lr=train_args["lr"])
    h_init = init_hidden(x.shape[0])

    # -------- Train

    losses, ckpts = rnns.fit_rnn(
        rnn,
        x,
        y,
        opt,
        n_steps=train_args["steps"],
        h_init=h_init,
        return_h=False,
        return_preds=False,
        device=device,
        session_batch=train_args["batch"],
        checkpoint_every=train_args["checkpt"],
    )

    # -------- Save

    rnn_hash = timehash(unique_within=rnn_root, ext=".pt")
    rnn_path = rnn_root / rnn_path_fmt.format(hash=rnn_hash)
    rnns.save_driscoll_rnn(
        rnn_path,
        rnn,
        dset_hash,
        task_hash,
        checkpoints=ckpts,
        losses=losses,
        source_meta={
            "source": "driscoll-multi-rnn.py",
            "network_type": network_type,
            "network_args": network_args,
            "train_args": train_args,
        },
    )
    print(f"Saved RNN model: {rnn_path}")
