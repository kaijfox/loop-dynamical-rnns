import torch as th
import torch.jit as jit
from torch import nn
import numpy as np
import copy
import dill
from collections import namedtuple
from .rnntasks import DriscollTasks, DriscollPlots
from .basic_rnns import plot_rnn_training
from sklearn.decomposition import PCA
import tqdm
import matplotlib.pyplot as plt
from mplutil.nb import colorset
from typing import Optional
from mplutil import util as vu

split_dataset = namedtuple("dataset", ["train", "val"])
activity_dataset = namedtuple(
    "activity_dataset",
    [
        "stim",
        "targets",
        "periods",
        "activity",
        "n_stim",
        "n_target",
        "n_period",
        "n_act",
        "session_length",
        "n_session",
        "metadata",
    ],
)
activity_dataset.__doc__ = """
A dataset of activity from a forced (potentially trained) dynamical system.

Properties
----------
stim : np.ndarray, shape (n_session, session_length, n_stim)
    The stimuli or forces presented to the system.
targets : np.ndarray, shape (n_session, session_length, n_target)
    The target outputs of the system if it was trained.
periods : np.ndarray of int, shape (n_session, session_length)
    Flag indicating the world state at each time step.
activity : np.ndarray, shape (n_session, session_length, n_act)
    The dynamical variablle of the system at each time step.
n_stim : int
    The number of stimulus / force dimensions.
n_target : int
    The number of target dimensions.
n_period : int
    The number of world states.
n_act : int
    The number of dynamical variables.
session_length : int
    The number of time steps in each session.
n_session : int
    The number of sessions.
metadata : dict
    Metadata or keyword arguments for to generate the stimuli and targets.
"""


class MultiBlockActivityDataset(DriscollTasks.MultiTaskBlockDataset):
    activity: np.ndarray
    n_act: int
    pca: Optional[PCA]


def save_dsn(
    model_path,
    model,
    dataset,
    gamma,
    cumulant_fn,
    checkpoints=None,
    losses=None,
    source_meta={},
):
    """
    Write three files to disk:
    {model_path}.tar:
        State dictionary of `model` and state dictionaries of each model in
        `checkpoints`.
    {model_path}.pt:
        Serialized `model`.
    {model_path}.train.dil
        Dataset object containing task context and activity data that the model
        was trained on.
    """
    th.save(
        {
            "state_dict": model.state_dict(),
            "checkpoints": (
                {i: m.state_dict() for i, m in checkpoints.items()}
                if checkpoints is not None
                else checkpoints
            ),
        },
        f"{model_path}.tar",
    )
    jit.save(model, f"{model_path}.pt")
    dill.dump(
        {
            "dataset": dataset,
            "cumulant_fn": cumulant_fn,
            "losses": losses,
            "gamma": gamma,
            **source_meta,
        },
        open(f"{model_path}.train.dil", "wb"),
    )


def load_dsn(model_path, device=None):
    """
    Load model serialized by by `save_dsn`.

    Parameters
    ----------
    model_path : str
        Path to the model file, without the extension, or with extension '.pt'.

    Returns
    -------
    model : nn.Module
    ckpts : dict of nn.Module
    train_data : dict
        Contining
        - `'dataset'`: task context and activity data that the model was trained
          on.
        - `'cumulant_fn'`: function for generating cumulant from dataset.train
          or dataset.val that was used to train the network.
        - `'losses'`: Loss values achieved over training.
        - `'gamma'`: Discount value.
    """
    model_path = str(model_path)
    if model_path.endswith(".pt"):
        model_path = model_path[:-3]
    model = jit.load(f"{model_path}.pt", map_location=device)
    params = th.load(f"{model_path}.tar", map_location=device)
    model.load_state_dict(params["state_dict"])
    ckpts = {i: copy.deepcopy(model) for i in params["checkpoints"]}
    for i, c in params["checkpoints"].items():
        ckpts[i].load_state_dict(c)
    train_data = dill.load(open(f"{model_path}.train.dil", "rb"))
    return model, ckpts, train_data


def fit_dsn(
    nn,
    x,
    y,
    opt,
    gamma,
    loss_fn=nn.MSELoss(),
    n_steps=2000,
    checkpoint_every=None,
):
    """
    Fit a network to calculate discounted sums.

    Minimize the error `td_loss(nn(x), y, gamma, loss_fn)`.

    Parameters
    ----------
    rnn : nn.Module
        The RNN model. Should have a function `seq_forward` that takes an input
        tensor and an initial hidden state tensor. And a function `init_hidden`
        that returns an initial hidden state tensor given a batch size.
    x : th.Tensor
        The input tensor. Should have shape (n_batch, n_time, n_dim).
    y : th.Tensor
        The target tensor. Should have shape (n_batch, n_time - 1, n_dim).
    opt : optim.Optimizer
        The optimizer to use.
    gamma : float
        Discount rate.
    loss_fn : nn.Module
        The loss function to use.
    n_steps : int
        The number of optimization steps to take.
    checkpoint_every : int
        Return copies of the model from every `checkpoint_every` epochs.

    Returns
    -------
    losses : np.ndarray
        The loss values at each epoch.
    models : List(nn.Module)
        The model at each checkpoint. Only returned if `checkpoint_every` is not None.
    """

    losses = []
    models = {}

    for i in tqdm.trange(n_steps):
        opt.zero_grad()
        loss = td_loss(nn(x), y, gamma, loss_fn)
        loss.backward()
        opt.step()

        losses.append(loss.detach().cpu().numpy())
        if checkpoint_every is not None and i % checkpoint_every == 0:
            models[i] = copy.deepcopy(nn).cpu()

    ret = (np.array(losses),)
    if checkpoint_every is not None:
        ret += (models,)
    return ret


def td_loss(prediction, cumulant, gamma, loss_fn=nn.MSELoss()):
    """
    For a black box differentiable function F(x) and sequence data
    $x_t\in\mathbb{R}^n$, $y_t\in\mathbb{R}^m$, minimize the objective
    $L = ||y_{t->t+1} + gamma * F(x_{t+1}) - F(x_t)||^2$
    if `loss_fn` is nn.MSELoss(), or more generally minimize
    $L = loss_fn(y_{t+1} + gamma * F(x_{t+1}), F(x_t)).$

    Parameters
    ----------
    prediction : shape (..., n_time, n_dim)
        Predictions of discounted future cumulant, $F(x)$.
    cumulant : shape (..., n_time - 1, n_dim)
        Cumulant (e.g., reward) term in target Bellman equation, $y$.
    """
    Fx_tplus1 = prediction[..., 1:, :]
    Fx_t = prediction[..., :-1, :]
    return loss_fn(cumulant + gamma * Fx_tplus1, Fx_t)


def discounted_sums(x, gamma):
    """
    Parameters
    ----------
    x : array, shape (.., n_samples)
    gamma : float
    """
    if th.is_tensor(x):
        m = th
    else:
        m = np
    T = x.shape[-1]
    gamma = gamma ** m.arange(T)
    return m.stack(
        [(x[..., i:] * gamma[: T - i]).sum(axis=-1) for i in range(T)], axis=-1
    )


def create_memorypro_activity_dataset(
    model,
    n_dim=40,
    n_train=200,
    n_val=50,
    seed_train=1,
    seed_val=2,
    session_length=500,
    apply_pca=True,
    task_kws={},
):
    """
    Generate embeddings of hidden trajectories from `memorypro` task.


    """

    # generate task stimuli for training and validation sessions
    x, y, periods = DriscollTasks.memorypro(
        **{
            **task_kws,
            **dict(
                session_length=session_length,
                n_sessions=n_train,
                seed=seed_train,
            ),
        }
    )
    x = th.tensor(x, dtype=th.float32)
    y = th.tensor(y, dtype=th.float32)

    xval, yval, periodsval = DriscollTasks.memorypro(
        **{
            **task_kws,
            **dict(
                session_length=500,
                n_sessions=n_val,
                seed=seed_val,
            ),
        }
    )
    xv = th.tensor(xval, dtype=th.float32)
    yv = th.tensor(yval, dtype=th.float32)

    # run model on stimuli x and xv to generate hidden trajectories
    device = next(model.parameters()).device
    h_init = th.zeros(x.shape[0], model.nh)
    _, h = model.seq_forward(x.to(device), h_init.to(device))

    h_init = th.zeros(xv.shape[0], model.nh)
    _, hv = model.seq_forward(xv.to(device), h_init.to(device))

    # compress hidden trajectories
    # (we really only care about leaning the first few PCs)
    if n_dim > 0:
        pca = PCA(n_components=n_dim)
        xh = pca.fit_transform(
            h.detach().cpu().numpy().reshape(-1, h.shape[-1])
        ).reshape(h.shape[:-1] + (n_dim,))
        xh_val = pca.transform(
            hv.detach().cpu().numpy().reshape(-1, h.shape[-1])
        ).reshape(hv.shape[:-1] + (n_dim,))
    else:
        pca = None
        xh = h.detach().cpu().numpy()
        xh_val = hv.detach().cpu().numpy()
        n_dim = xh.shape[-1]
    xh = th.tensor(xh, dtype=th.float32)
    xh_val = th.tensor(xh_val, dtype=th.float32)

    meta = dict(
        n_stim=x.shape[-1],
        n_target=y.shape[-1],
        n_period=periods.max() + 1,
        n_act=n_dim,
        session_length=x.shape[1],
        metadata={"task_kws": task_kws, "pca": pca},
    )
    return split_dataset(
        train=activity_dataset(
            stim=x,
            targets=y,
            periods=periods,
            activity=xh,
            n_session=x.shape[0],
            **meta,
        ),
        val=activity_dataset(
            stim=xv,
            targets=yv,
            periods=periodsval,
            activity=xh_val,
            n_session=xv.shape[0],
            **meta,
        ),
    )


def evaluate_multiblock_activity(
    rnn, pca_set: str = None, n_dim: int = 40, apply_pca: bool = True, **kws
):
    """
    Generate embeddings of hidden trajectories from `memorypro` task.

    Parameters
    ----------
    rnn : nn.Module
        The RNN model. Should have a function `seq_forward` taking input and
        initial hidden state tensors.
    n_dim : int
        Number of dimensions to reduce the hidden state to. A nonpositive value
        indicated no reduction.
    apply_pca : bool
        Whether to apply PCA to the hidden state, assumed true if `n_dim` is
        positive.
    pca_set : str
        Which dataset to use for training PCA.
    kws : dict[str, DriscollTasks.MultiTaskBlockDataset]
        Datasets to generate activity for.
    """

    sets = list(kws.keys())
    sets = [pca_set] + [s for s in sets if s != pca_set]
    pca = None
    ret = {}
    for set_name in sets:
        dataset: DriscollTasks.MultiTaskBlockDataset = kws[set_name]
        print("Applying to set", set_name)

        # run model on stimuli to generate hidden trajectories
        x = th.tensor(dataset["stimuli"], dtype=th.float32)
        device = next(rnn.parameters()).device
        h_init = th.zeros(x.shape[0], rnn.nh)
        _, h = rnn.seq_forward(x.to(device), h_init.to(device))

        # optionally compress hidden trajectories
        apply_pca = apply_pca or n_dim > 0
        if apply_pca:
            if set_name == pca_set:
                if n_dim < 0:
                    n_dim = h.shape[-1]
                print(f"fitting pca to {n_dim} components")
                pca = PCA(n_components=n_dim)
                h_reduced = pca.fit_transform(
                    h.detach().cpu().numpy().reshape(-1, h.shape[-1])
                ).reshape(h.shape[:-1] + (n_dim,))
            else:
                h_reduced = pca.transform(
                    h.detach().cpu().numpy().reshape(-1, h.shape[-1])
                ).reshape(h.shape[:-1] + (n_dim,))
        else:
            h_reduced = h.detach().cpu().numpy()
            n_dim = h.shape[-1]

        # Embed in dataset structure and return
        ret[set_name] = MultiBlockActivityDataset(
            **dataset, activity=h_reduced, n_act=n_dim, pca=pca
        )
    return ret


def plot_memorypro_prediction(
    losses,
    dataset,
    predictions,
    targets,
    losses_val,
    plot_units,
    xcolors=None,
    ycolors=None,
    session=0,
    colors=None,
):
    if xcolors is None:
        xcolors = DriscollPlots.memorypro_xcolors
    if ycolors is None:
        ycolors = DriscollPlots.memorypro_ycolors
    if colors is None:
        colors = colorset.active

    n_units = len(plot_units)
    epochs = sorted(list(predictions.keys()))
    # predictions = [predictions[i] for i in epochs]
    losses_val = [losses_val[i] for i in epochs]

    # plot training losses and predictions
    fig, ax = plt.subplots(
        2,
        n_units + 1,
        figsize=(2 + 3 * n_units, 2),
        sharex="col",
        width_ratios=[1] + [2] * n_units,
    )
    plot_rnn_training(
        losses,
        predictions,
        targets,
        epochs=epochs,
        session=session,
        ax=ax[0, :],
    )

    ax[0, 0].set_yscale("log")
    ax[1, 0].set_axis_off()
    ax[0, 0].plot(epochs, losses_val, color=colors.neutral, zorder=-1)

    # plot task context
    for r in range(2):
        for i in range(0, n_units):
            DriscollPlots.memorypro(
                ax[r, i + 1],
                dataset.stim if r == 1 else None,
                dataset.targets if r == 1 else None,
                dataset.periods,
                session=session,
                xcolors=xcolors,
                ycolors=ycolors,
                single_ax=True,
                flags=False,
            )
            if r == 1:
                ax[r, i + 1].twinx().plot(
                    dataset.activity[session, :, plot_units[i]],
                    color=colors.neutral,
                    lw=1,
                )

    return fig, ax


def plot_memorypro_dsn_prediction(
    losses,
    val_dataset: activity_dataset,
    cumulant: th.tensor,
    gamma: float,
    checkpoints: dict,
    plot_units: list,
    loss_fn=nn.MSELoss(),
    session=0,
):
    # evaluate each model checkpoint
    steps = checkpoints.keys()
    with th.no_grad():
        ckpt_yhats = {
            i: checkpoints[i].cpu()(val_dataset.activity.cpu()).detach() for i in steps
        }
    ckpt_preds = {i: y[..., plot_units].numpy() for i, y in ckpt_yhats.items()}
    ckpt_loss = {i: td_loss(y, cumulant, gamma, loss_fn) for i, y in ckpt_yhats.items()}
    manual_sums = discounted_sums(
        cumulant[..., plot_units].transpose(2, 1), gamma
    ).transpose(2, 1)

    fig, ax = plot_memorypro_prediction(
        losses,
        val_dataset,
        ckpt_preds,
        manual_sums,
        ckpt_loss,
        plot_units,
        session=session,
    )
    ax[0, 0].set_yscale("log")

    return fig, ax


def create_predictor_network(net_type, net_args, n_act):
    """
    Generate a function returning an nn.Module for predicting discounted sums.

    The valid network types and their parameters are:
    - 'softplus': a feedforward network with softplus activations and layer
      normalization.
        - widths, list of two ints: the number of hidden units in nonlinear
          layer
    - 'bottle-softplus': a feedforward network with softplus activations and layer
        normalization, and an initial linear bottleneck.
        - widths, list of two ints: the number of hidden units in nonlinear layer
        - bottleneck, int: the number of units in the bottleneck layer

    Parameters
    ----------
    net_type : str
        The type of network to generate. Options are:
        - 'softplus': a feedforward network with softplus activations.
    net_args : dict
        Arguments to pass to the network function, described above.
    n_act : int
        The number of input and output dimensions.
    """

    if net_type == "bottle-softplus":
        net_args = {**dict(widths=[], bottleneck=-1), **net_args}
        widths = net_args["widths"]
        if len(widths) != 2:
            raise ValueError(
                f"`widths` network argument must" " be a list of length 2, got {widths}"
            )
        if net_args["bottleneck"] < 0:
            raise ValueError(f"`bottleneck` network argument required")
        create_model = lambda: nn.Sequential(
            nn.Linear(n_act, net_args["bottleneck"]),
            nn.Linear(net_args["bottleneck"], widths[0]),
            nn.Softplus(),
            nn.LayerNorm(widths[0]),
            nn.Linear(widths[0], widths[1]),
            nn.Softplus(),
            nn.Linear(widths[1], n_act),
        )

    elif net_type == "softplus":
        net_args = {
            **dict(
                widths=[],
            ),
            **net_args,
        }
        widths = net_args["widths"]
        if len(widths) != 2:
            raise ValueError(
                f"`widths` network argument must" " be a list of length 2, got {widths}"
            )
        create_model = lambda: nn.Sequential(
            nn.Linear(n_act, widths[0]),
            nn.Softplus(),
            nn.LayerNorm(widths[0]),
            nn.Linear(widths[0], widths[1]),
            nn.Softplus(),
            nn.Linear(widths[1], n_act),
        )

    else:
        raise ValueError(f"Invalid network type: {net_type}")

    return create_model


def plot_block_error_examples(
    test_data, cumul_gt, test_losses, test_preds, n_unit=2, session=0
):
    """
    Parameters
    ----------
    test_data : DriscollTasks.MultiTaskBlockDataset
    test_losses : dict[int, array shape (n_sess, session_len, n_act)]
    test_preds : dict[int, array shape (n_sess, session_len, n_act)]
    """
    n_blocks = len(test_data["block_slices"])
    block_data = DriscollTasks.split_dataset(test_data)

    bigfig, figs, _ = vu.flat_subfig_grid(n_blocks, 5, (4, (n_unit + 1) * 0.75))
    for ibl in range(n_blocks):
        fig = figs[ibl]
        task: DriscollTasks.DriscollTask = block_data[ibl]["task"]
        block: DriscollTasks.SingleTaskDataset = block_data[ibl]
        steps = np.array(sorted(list(test_preds.keys())))
        plot_step = steps[-1]
        _, ax, _ = vu.flat_grid((n_unit + 1), 1, ax_size=None, fig=fig, sharex=True)

        DriscollTasks.plot_session(
            task, periods=block["periods"], session=session, ax=ax
        )
        x = np.arange(1, block["session_length"])
        slc = test_data["block_slices"][ibl]
        block_loss = test_losses[plot_step][slc]
        block_preds = test_preds[plot_step][slc]
        ax[0].plot(
            x,
            block_loss[session].mean(axis=-1),
            color="C0",
            lw=1,
        )
        for i in range(n_unit):
            ax[i + 1].plot(
                x,
                block_preds[session, :, i],
                color="C3",
                lw=1,
            )
            ax[i + 1].plot(
                x,
                cumul_gt[session, :, i],
                color="k",
                lw=1,
            )

    return bigfig
