import torch as th
from torch import jit
import numpy as np
import joblib as jl
import copy
import glob
from torch import nn
import matplotlib.pyplot as plt
import time
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable
import scipy.stats
import tqdm
import dill

from .viz import styles


def _dict_to(d, device):
    """
    Recursively apply t.to(device) to all tensors in d.
    """
    if th.is_tensor(d):
        return d.to(device)
    if hasattr(d, 'items'):
        return {
            k: _dict_to(v, device) for k, v in d.items()
        }
    if isinstance(d, Iterable):
        return [_dict_to(v, device) for v in d]
    return d
    

class SignedLinear(nn.Module):
    def __init__(self, n, n_out=None, sign=1, scale=1, allow_diag=True):
        super().__init__()
        self.n_in = n
        self.sign = sign
        self.scale = scale

        self.n_out = n if n_out is None else n_out
        if not allow_diag:
            self.mask = 1 - th.eye(n)
            self.n_out = n
        else:
            self.mask = 1

        self.weight = nn.Parameter(th.Tensor(self.n_out, self.n_in))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        with th.no_grad():
            self.weight.set_(th.sqrt(th.abs(self.weight)) / 2)

    def forward(self, input):
        # w = self.scale * (self.weight ** 2) * self.mask
        return input @ self._weight().T

    def _weight(self):
        return (self.sign * self.scale) * (self.weight**2) * self.mask


class LowRankLinear(nn.Module):
    def __init__(self, n, n_out=None, rank=1, bias=False, init="xavier"):
        super().__init__()
        self.n_in = n
        self.n_out = n if n_out is None else n_out
        self.rank = rank
        self.init = init
        assert self.init in [
            "xavier",
            "ortho-inv",
        ], "Only 'xavier' and 'ortho-inv' supported."

        self.v = nn.Parameter(th.Tensor(self.n_in, self.rank))
        self.u = nn.Parameter(th.Tensor(self.n_out, self.rank))
        if bias:
            self.bias = nn.Parameter(th.Tensor(self.n_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init == "xavier":
            W = th.empty(self.n_out, self.n_in)
            nn.init.xavier_uniform_(W)
            with th.no_grad():
                u, s, v = th.svd(W)
                # truncate to self.rank
                self.u.set_(u[:, : self.rank])
                self.v.set_(v[:, : self.rank])
        elif self.init == 'ortho-inv':
            with th.no_grad():
                s_ = th.rand(())
                seed = abs(int(s_ * (2 ** 32 - 1)))
                v = scipy.stats.ortho_group.rvs(self.n_in, random_state=seed)
                self.v.set_(th.tensor(v[:, :self.rank], dtype=th.float32))
                W = scipy.stats.ortho_group.rvs(self.rank, random_state=seed+1)
                u = v[:, :self.rank] @ W
                self.u.set_(th.tensor(u, dtype=th.float32))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._weight())
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Parameters
        ----------
        input : torch.Tensor, (n_batch, n_in)

        Returns
        -------
        torch.Tensor, (n_batch, n_out)

        """
        # W = self._weight()
        # return nn.functional.linear(input, W, self.bias)
        return (input @ self.v) @ self.u.T + self.bias

    def _weight(self):
        """
        Returns
        -------
        torch.Tensor, (n_out, n_in)
        """
        return self.u @ self.v.T

    def extra_repr(self) -> str:
        return (
            f"n_in={self.n_in}, n_out={self.n_out},"
            "bias={self.bias is not None} rank={self.rank}"
        )


def init_dynamical_rnn(
    self, nx, nh, ny=None, alpha=0.1, act=nn.Sigmoid(), h_bias=0, w_scale=1, act_ofs=0
):
    if ny is None:
        ny = nx

    self.nx = nx
    self.nh = nh
    self.ny = ny

    self.act = act
    self.alpha = alpha
    self.h_bias = h_bias
    self.w_scale = w_scale
    self.act_ofs = act_ofs


class DynamicalRNN(nn.Module):
    def __init__(self, nx=4, nh=10, ny=None):
        super().__init__()
        if ny is None:
            ny = nx

        self.nx = nx
        self.nh = nh
        self.ny = ny

        self.i2h = nn.Linear(nx, nh, bias=False)
        self.h2h = nn.Linear(nh, nh, bias=False)
        self.h2y = nn.Linear(nh, ny, bias=False)
        self.act = nn.Tanh()

    def forward(self, x, h):
        h = self.act(self.i2h(x) + self.h2h(h))
        y = self.h2y(h)
        return y, h

    def seq_forward(self, x, h):
        """
        Forward pass for a sequence of inputs.

        Parameters
        ----------
        x : torch.Tensor, (batch_size, hidden_size)
            The input sequence.
        h : torch.Tensor, (batch_size, sequence_length, input_size)
            The initial hidden state tensor.

        Returns
        -------
        outputs : torch.Tensor, (batch_size, sequence_length, output_size).
            Sequence of output unit activations
        hidden_states : torch.Tensor, (batch_size, hidden_size)
            Sequence of hidden state activations.
        """
        y = []
        hs = []
        for i in range(x.shape[1]):
            y_, h = self.forward(x[:, i], h)
            y.append(y_)
            hs.append(h)
        return th.stack(y, dim=1), th.stack(hs, dim=1)

    def init_hidden(self, batch_size, device=None):
        return th.zeros(batch_size, self.nh, device=device)

    def init_weights(self):
        pass


class NegRNN(DynamicalRNN):
    def __init__(
        self,
        nx=4,
        nh=10,
        ny=None,
        alpha=0.1,
        act=nn.Sigmoid(),
        h_bias=0,
        w_scale=1,
        act_ofs=0,
    ):
        """
        Approximate dynamical RNN with negative weights.

        Parameters
        ----------
        alpha : float
            Step size divided by time constant, or equivalently a coefficient
            for convex combination of (in [0, 1]) $h_{t-1}$ and $f(h_{t-1})$ in
            the hidden state update rule, with `alpha` equal to 1 corresponding
            to a no-memory update.
        """
        nn.Module.__init__(self)
        init_dynamical_rnn(self, nx, nh, ny, alpha, act, h_bias, w_scale, act_ofs)

        self.i2h = nn.Linear(self.nx, self.nh, bias=False)
        self.h2h = SignedLinear(
            self.nh, sign=-1, allow_diag=False
        )  # nn.Linear(nh, nh, bias = False)
        self.h2y = nn.Linear(self.nh, self.ny, bias=False)

    def forward(self, x, h):
        h_act = self.act(h + self.h_bias) + self.act_ofs
        fh = self.w_scale * self.h2h(h_act) + self.i2h(x)
        h_new = (1 - self.alpha) * h + self.alpha * fh
        y = self.h2y(h_new)
        return y, h_new


class BasicRNN(NegRNN):

    def __init__(
        self,
        nx,
        nh,
        ny=None,
        alpha=0.1,
        act=nn.Sigmoid(),
        h_bias=0,
        w_scale=1,
        act_ofs=0,
        bias=False,
    ):
        nn.Module.__init__(self)
        init_dynamical_rnn(self, nx, nh, ny, alpha, act, h_bias, w_scale, act_ofs)

        self.i2h = nn.Linear(self.nx, self.nh, bias=bias)
        self.h2h = nn.Linear(self.nh, self.nh, bias=bias)
        self.h2y = nn.Linear(self.nh, self.ny, bias=bias)


class BasicRNN_LN(nn.Module):
    def __init__(
        self,
        nx=4,
        nh=10,
        ny=None,
        alpha=0.1,
        act=nn.Sigmoid(),
        h_bias=0,
        w_scale=1,
        act_ofs=0,
        bias=False,
    ):
        """
        Approximate dynamical RNN with negative weights.

        Parameters
        ----------
        alpha : float
            Step size divided by time constant, or equivalently a coefficient
            for convex combination of (in [0, 1]) $h_{t-1}$ and $f(h_{t-1})$ in
            the hidden state update rule, with `alpha` equal to 1 corresponding
            to a no-memory update.
        """
        nn.Module.__init__(self)
        init_dynamical_rnn(self, nx, nh, ny, alpha, act, h_bias, w_scale, act_ofs)

        self.i2h = nn.Linear(self.nx, self.nh, bias=bias)
        self.h2h = nn.Linear(self.nh, self.nh, bias=bias)
        self.h2y = nn.Linear(self.nh, self.ny, bias=bias)
        self.hbn = nn.LayerNorm(self.nh, elementwise_affine=False)

    @jit.export
    def forward(self, x, h):
        h_norm = self.hbn(h)
        I = self.w_scale * self.h2h(h_norm) + self.i2h(x)
        fh = self.act(I + self.h_bias) + self.act_ofs
        h_new = (1 - self.alpha) * h + self.alpha * fh
        y = self.h2y(h_new)
        return y, h_new

    @jit.export
    def seq_forward(self, x, h):
        y = []
        hs = []
        for i in range(x.shape[1]):
            y_, h = self.forward(x[:, i], h)
            y.append(y_)
            hs.append(h)
        return th.stack(y, dim=1), th.stack(hs, dim=1)

    def init_hidden(self, batch_size, device=None):
        return th.zeros(batch_size, self.nh, device=device)

    def init_weights(self):
        pass


class BasicRNN_LR(nn.Module):
    def __init__(
        self,
        nx=4,
        nh=10,
        ny=None,
        alpha=0.1,
        act=nn.Sigmoid(),
        h_bias=0,
        w_scale=1,
        act_ofs=0,
        bias=False,
        rank=1,
        init="xavier",
    ):
        """

        Parameters
        ----------
        alpha : float
            Step size divided by time constant, or equivalently a coefficient
            for convex combination of (in [0, 1]) $h_{t-1}$ and $f(h_{t-1})$ in
            the hidden state update rule, with `alpha` equal to 1 corresponding
            to a no-memory update. Only alpha == 1 supported.
        h_bias : float
            Bias to add to hidden state before activation (inverse threshold of
            hidden state activation function). Only 0 supported.
        w_scale : float
            Scale of the weight matrix. Only 1 supported.
        act_ofs : float
            Constant to add to hidden state activation function (i.e. after
            nonlinearity, base firing rate). Only 0 supported.
        bias : bool
            Whether to include bias terms in the linear layers.
        rank : int
            The rank of the weight matrix.
        init : str
            The initialization method for the weight matrix. Only 'xavier'
            and 'ortho-inv' suported. 'ortho' set the weight matrix to a
            low-rank truncated Xavier initialization. 'ortho-inv' chooses
            random orthogonal decoder vectors (v.T) and sets encoder vectors (u)
            such that $v.t @ u

        """
        nn.Module.__init__(self)
        init_dynamical_rnn(self, nx, nh, ny, alpha, act, h_bias, w_scale, act_ofs)
        assert self.h_bias == 0, "Only h_bias == 0 supported."
        assert self.w_scale == 1, "Only w_scale == 1 supported."
        assert self.act_ofs == 0, "Only act_ofs == 0 supported."
        self.rank = int(rank)
        self.init = init

        self.i2h = nn.Linear(self.nx, self.nh, bias=bias)
        self.h2h = LowRankLinear(self.nh, self.nh, self.rank, bias=bias, init=self.init)
        self.h2y = nn.Linear(self.nh, self.ny, bias=bias)
        self.hbn = nn.LayerNorm(self.nh, elementwise_affine=False)

    @jit.export
    def forward(self, x, h):
        h_norm = self.hbn(h)
        I = self.h2h(h_norm) + self.i2h(x)
        fh = self.act(I)
        h_new = (1 - self.alpha) * h + self.alpha * fh
        y = self.h2y(h_new)
        return y, h_new

    @jit.export
    def seq_forward(self, x, h):
        y = []
        hs = []
        for i in range(x.shape[1]):
            y_, h = self.forward(x[:, i], h)
            y.append(y_)
            hs.append(h)
        return th.stack(y, dim=1), th.stack(hs, dim=1)

    def init_hidden(self, batch_size, device=None):
        return th.zeros(batch_size, self.nh, device=device)

    def init_weights(self):
        pass


### ------------------------------------------------- Fitting and plotting ----


def fit_rnn(
    rnn,
    x,
    y,
    opt,
    loss_fn=nn.MSELoss(),
    h_init=None,
    n_steps=2000,
    lr=None,
    device=None,
    session_batch=None,
    batch_seed=None,
    checkpoint_every=None,
    save_fn=None,
    save_every=None,
    loss_every=None,
    first_step=0,
):
    """
    Fit an RNN to batched sequences.

    Parameters
    ----------
    rnn : nn.Module
        The RNN model. Should have a function `seq_forward` that takes an input
        tensor and an initial hidden state tensor. And a function `init_hidden`
        that returns an initial hidden state tensor given a batch size.
    x : th.Tensor
        The input tensor. Should have shape (n_batch, n_step, n_inputs).
    y : th.Tensor
        The target tensor. Should have shape (n_batch, n_step, n_outputs).
    opt : optim.Optimizer
        The optimizer to use.
    loss_fn : nn.Module
        The loss function to use.
    h_init : th.Tensor
        The initial hidden state tensor. If None, it is initialized by the RNN.
    n_steps : int
        The number of optimization steps to take.
    lr : object
        `torch.optim` learning rate scheduler.
    device : str or th.device, optional
        If given a device to move training data before running the network. Will
        be moved each step / batch, so if not batching it is better to manually
        move the manually before passing.
    session_batch : int
        If given, the number of sessions to train on in each step. This is
        useful for training on large datasets that do not fit in GPU memory.
    batch_seed : int
        The seed to use for sampling batches.
    checkpoint_every : int
        Return copies of the model from every `checkpoint_every` epochs.

    Returns
    -------
    losses : np.ndarray
        The loss values at each epoch.
    yhats : np.ndarray
        The predicted output values at each epoch. Returned only if
        `return_preds` is True.
    h_hist : list[np.ndarray]
        The hidden state values at each epoch, returned only if `return_h` is
        True. Not stacked because this can crash the kernel for long trainings
        of large networks.
    lr_hist : np.ndarray
        The learning rate values at each epoch, returned only if `lr` is passed.
    ckpts : list[nn.Module]
        The model copies at each checkpoint, returned only if `checkpoint_every`
        is passed.
    """

    lr_hist = []
    ckpts = {}
    rng = np.random.default_rng(batch_seed)
    if loss_every is not None:
        losses = np.full([2, n_steps // loss_every + 1], np.nan)

    if h_init is None:
        h_init = rnn.init_hidden(x.shape[0], device=x.device)
    for step in tqdm.trange(first_step, n_steps + first_step):
        i = step - first_step

        if session_batch is not None:
            idx = rng.choice(x.shape[0], session_batch, replace=False).tolist()
            x_ = x[idx]
            y_ = y[idx]
            h_init_ = h_init[idx]
        else:
            x_ = x
            y_ = y
            h_init_ = h_init
        if device is not None:
            x_ = x_.to(device)
            y_ = y_.to(device)
            h_init_ = h_init_.to(device)

        opt.zero_grad()
        yhat, hs = rnn.seq_forward(x_, h_init_)
        loss = loss_fn(yhat, y_)
        loss.backward()
        opt.step()
        if loss_every is not None and i % loss_every == 0:
            losses[0, i // loss_every] = loss.detach().cpu().numpy()
            losses[1, i // loss_every] = step
        if lr is not None:
            lr.step()
            lr_hist.append(opt.param_groups[0]["lr"])
        if checkpoint_every is not None and i % checkpoint_every == 0:
            ckpts[step] = copy.deepcopy(rnn).cpu()
        if save_fn is not None and i % save_every == 0:
            save_fn(
                {
                    "model": rnn,
                    "losses": losses[:, : i // loss_every],
                    "checkpoints": ckpts,
                    "step": step,
                }
            )

    ret = ((losses),)
    if lr is not None:
        ret += (np.array(lr_hist),)
    if checkpoint_every is not None:
        ret += (ckpts,)
    return ret


def fit_ffn(
    nn,
    x,
    y,
    opt,
    loss_fn=nn.MSELoss(),
    n_steps=2000,
    lr=None,
    return_batches=None,
    checkpoint_every=None,
):
    """
    Fit a feedforward neural network.

    Parameters
    ----------
    rnn : nn.Module
        The RNN model. Should have a function `seq_forward` that takes an input
        tensor and an initial hidden state tensor. And a function `init_hidden`
        that returns an initial hidden state tensor given a batch size.
    x : th.Tensor
        The input tensor. Should have shape (n_batch, n_inputs).
    y : th.Tensor
        The target tensor. Should have shape (n_batch, n_outputs).
    opt : optim.Optimizer
        The optimizer to use.
    loss_fn : nn.Module
        The loss function to use.
    n_steps : int
        The number of optimization steps to take.
    lr : object
        `torch.optim` learning rate scheduler.
    return_batches : list[int] or callable[[array], array]
        The indices of the sessions for which to return the predicted values or
        a function that takes all predicted values and returns those to save.
    checkpoint_every : int
        Return copies of the model from every `checkpoint_every` epochs.

    Returns
    -------
    losses : np.ndarray
        The loss values at each epoch.
    yhats : list[np.ndarray]
        The predicted output values at each epoch. These are not stacked because
        they can be large.
    lr_hist : np.ndarray
        The learning rate values at each epoch, returned only if `lr` is passed.
    """

    losses = []
    yhats = []
    lr_hist = []
    models = []

    for i in tqdm.trange(n_steps):
        opt.zero_grad()
        yhat = nn(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        opt.step()

        losses.append(loss.detach().cpu().numpy())
        if return_batches is not None:
            if isinstance(return_batches, list):
                yhats.append(yhat.detach().cpu().numpy()[return_batches])
            else:
                yhats.append(return_batches(yhat.detach().cpu().numpy()))
        if lr is not None:
            lr.step()
            lr_hist.append(opt.param_groups[0]["lr"])
        if checkpoint_every is not None and i % checkpoint_every == 0:
            models.append(copy.deepcopy(nn).cpu())

    ret = (np.array(losses), yhats)
    if lr is not None:
        ret += (np.array(lr_hist),)
    if checkpoint_every is not None:
        ret += (models,)
    return ret


def plot_rnn_training(
    losses,
    yhats,
    x,
    epochs=None,
    start=0,
    colors=None,
    n_iter=5,
    session=0,
    col_buffer=3,
    ex_epochs=None,
    loss_matches_epochs=False,
    lr=None,
    ax=None,
):
    skip = len(losses) // n_iter
    colors = styles.default(colors)

    lr_i = 0 if lr is None else 1
    if ax is None:
        fig, ax = plt.subplots(
            1, x.shape[-1] + 1 + lr_i, figsize=(2 * (x.shape[-1] + 1) + 2 * lr_i, 2)
        )
        ret = fig, ax
    else:
        ret = None

    if loss_matches_epochs:
        ax[0].plot(epochs, [losses[ep] for ep in epochs], color=colors.subtle)
    else:
        ax[0].plot(losses, color=colors.subtle)

    buf = col_buffer * skip
    if epochs is None:
        epochs = range(start, len(losses), skip)
    if ex_epochs is None:
        ex_epochs = epochs
    pal = colors.ch0(epochs)

    if th.is_tensor(x):
        x = x.numpy()

    for i in range(x.shape[-1]):
        ax[i + 1].plot(x[session, :, i], color=colors.subtle, zorder=2)
        for ep in epochs:
            ax[0].plot([ep], [losses[ep]], "o", ms=3, color=pal[ep])
        for ep in ex_epochs:
            ax[i + 1].plot(yhats[ep][session, :, i], color=pal[ep])

    if lr is not None:
        ax[-1].plot(lr, color=colors.neutral, lw=1)
        ax[-1].set_yscale("log")

    return ret


def timehash(unique_within=False, ext=".*", max_suffix=1000):
    """
    Parameters
    ----------
    unique_within, bool or str:
        If a string, the hash will be postfixed with a unique identifier if
        there is another file with the same hash.
    """
    hextime = str(hex(int(datetime.now().strftime("%m%d%H%M")))).lstrip("0x")
    # suffix with hex codes to ensure unique filenames
    if unique_within:
        i = 0
        unique_hextime = hextime
        while (
            find_hash(unique_within, unique_hextime, ext=ext, silent=True) is not None
            and i < max_suffix
        ):
            unique_hextime = f"{hextime}.{str(hex(i)).lstrip('0x')}"
            i += 1
        if i > max_suffix - 1:
            raise ValueError(
                "Could not find unique hash within "
                f"{max_suffix} attempts. Last attempt was {unique_hextime}"
            )
        hextime = unique_hextime
    return hextime


def find_hash(root, hash, ext=".*", silent=False):
    pattern = str(Path(root) / f"**/*_{hash}{ext}")
    results = list(glob.glob(pattern, recursive=True))
    if not len(results):
        if silent:
            return None
        raise ValueError(f"No maches for: {pattern}")
    return results[0]


def hash_or_path(path_str, ext=".*", sep=":"):
    """
    Resolve <root><sep><hash> or <path> format to filepath.

    Parameters
    ----------
    path_str : str
        The path spec string, either of format <root><sep><hash> or <path>.
    ext : str
        The extension to append to the hash in searching.
    sep : str
        The separator between the root and hash in the path spec string.
    """
    if sep in path_str:
        try:
            root, hash = path_str.split(":")
        except:
            raise ValueError(
                f"Path spec {path_str} contained separator but"
                " did not match pattern <root>{sep}<hash>"
            )
        return find_hash(root, hash, ext), hash
    return path_str, None


def save_rnn_deprecated(model_path, model, x, y, losses, yhats, meta={}):
    th.save(model.state_dict(), f"{model_path}.tar")
    jit.save(model, f"{model_path}.pt")
    jl.dump(
        {
            "x": x,
            "y": y,
            "meta": meta,
            "training": {"losses": losses, "yhats": yhats},
        },
        f"{model_path}.train.jl",
    )


def save_driscoll_rnn(
    model_path,
    model,
    dataset_hash,
    task_hash,
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
        Dataset object containing hashes for the task and dataset, as well as
        training losses if available and any other metadata.

    Parameters
    ----------
    model_path : str or Path
        The path to save the model to. If the path ends with `.pt`, the
        extension will be removed.
    model : nn.Module
        The model to save.
    dataset_hash : str
        The hash of the dataset used to train the model.
    task_hash : str
        The hash of the task used to train the model.
    checkpoints : dict[int, nn.Module], optional
        A dictionary of model checkpoints to save.
    losses : list[float], optional
        The training losses.
    source_meta : dict, optional
        Any additional metadata to save.
    """
    print("losses:", losses.shape)
    print("ckpts:", checkpoints.keys())
    print("step:", source_meta.get("step", None))
    # Allow referencing model path via .pt extension, instead of extensionless
    # format
    if str(model_path).endswith(".pt"):
        model_path = Path(str(model_path[:-3]))

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
            "dataset_hash": dataset_hash,
            "task_hash": task_hash,
            "losses": losses,
            **source_meta,
        },
        open(f"{model_path}.train.dil", "wb"),
    )


def load_rnn(model_path, aux=True, device=None):
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
        Only returned if `aux` is True. Root directories for hashes are assumed
        to be inferrable from context. Contiains:
        - `dataset_hash`, The hash of the dataset used to train the model.
        - `task_hash`, The hash of the task used to train the model.
        - `losses`, The training losses.
        - Any additional metadata passed to `save_dsn`.
    """
    model_path = str(model_path)
    # Allow referencing model path via .pt extension, instead of extensionless
    # format
    if model_path.endswith(".pt"):
        model_path = model_path[:-3]

    # --- Load model
    model = jit.load(f"{model_path}.pt", map_location=device)
    params = th.load(f"{model_path}.tar", map_location=device)
    model.load_state_dict(params["state_dict"])
    ckpts = {i: copy.deepcopy(model) for i in params["checkpoints"]}
    for i, c in params["checkpoints"].items():
        ckpts[i].load_state_dict(c)
    ret = (model, ckpts)
    # --- Load training auxiliary data
    if aux:
        train_data = dill.load(open(f"{model_path}.train.dil", "rb"))
        ret = ret + (train_data,)
    return ret
