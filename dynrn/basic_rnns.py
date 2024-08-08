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
import tqdm

from .viz import styles


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
        bias = False,
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
    
    def init_hidden(self, batch_size, device = None):
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
    return_h=True,
    lr=None,
    device=None,
    session_batch=None,
    batch_seed=None,
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
    return_h : bool
        Whether to return the hidden state values at each epoch.
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

    Returns
    -------
    losses : np.ndarray
        The loss values at each epoch.
    yhats : np.ndarray
        The predicted output values at each epoch.
    h_hist : list[np.ndarray]
        The hidden state values at each epoch, returned only if `return_h` is
        True. Not stacked because this can crash the kernel for long trainings
        of large networks.
    lr_hist : np.ndarray
        The learning rate values at each epoch, returned only if `lr` is passed.
    """

    losses = []
    yhats = []
    h_hist = []
    lr_hist = []
    rng = np.random.default_rng(batch_seed)

    if h_init is None:
        h_init = rnn.init_hidden(x.shape[0], device=x.device)
    for i in tqdm.trange(n_steps):
        if session_batch is not None:
            idx = rng.choice(x.shape[0], session_batch, replace=False).tolist()
            x_ = x[idx]
            y_ = y[idx]
            h_init_ = h_init[idx]
        else:
            x_ = x
            y_ = y
            h_init_ = h_init

        opt.zero_grad()
        yhat, hs = rnn.seq_forward(x_, h_init_)
        loss = loss_fn(yhat, y_)
        loss.backward()
        opt.step()
        losses.append(loss.detach().cpu().numpy())
        yhats.append(yhat.detach().cpu().numpy())
        if return_h:
            h_hist.append(hs.detach().cpu().numpy())
        if lr is not None:
            lr.step()
            lr_hist.append(opt.param_groups[0]["lr"])

    ret = (np.stack(losses), np.stack(yhats))
    if return_h:
        ret += (h_hist,)
    if lr is not None:
        ret += (np.array(lr_hist),)
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

    ax[0].plot(losses, color=colors.subtle)

    buf = col_buffer * skip
    if epochs is None:
        epochs = range(start, len(losses), skip)
    pal = colors.ch0(epochs)
    for i in range(x.shape[-1]):
        ax[i + 1].plot(x.numpy()[session, :, i], color=colors.subtle, zorder=2)
        for ep in epochs:
            ax[0].plot([ep], [losses[ep]], "o", ms=3, color=pal[ep])
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


def save_rnn(model_path, model, x, y, losses, yhats, meta={}):
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
