import torch as th
import numpy as np
from torch import optim
from torch import nn
from dynrn.rnntasks import SimpleTasks
from dynrn.viz import util as vu
import matplotlib.pyplot as plt
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

    def init_hidden(self, batch_size):
        return th.zeros(batch_size, self.nh)

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
        bias = False,
    ):
        nn.Module.__init__(self)
        init_dynamical_rnn(self, nx, nh, ny, alpha, act, h_bias, w_scale, act_ofs)

        self.i2h = nn.Linear(self.nx, self.nh, bias=bias)
        self.h2h = nn.Linear(self.nh, self.nh, bias=bias)
        self.h2y = nn.Linear(self.nh, self.ny, bias=bias)


### ------------------------------------------------- Fitting and plotting ----


def fit_rnn(rnn, x, y, opt, loss_fn=nn.MSELoss(), n_steps=2000):
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
    n_steps : int
        The number of optimization steps to take.
    """

    losses = []
    yhats = []
    h_hist = []

    for i in tqdm.trange(n_steps):
        opt.zero_grad()
        h_init = rnn.init_hidden(x.shape[0])
        yhat, hs = rnn.seq_forward(x, h_init)
        loss = loss_fn(yhat, y)
        loss.backward()
        opt.step()
        losses.append(loss.detach().numpy())
        yhats.append(yhat.detach().numpy())
        h_hist.append(hs.detach().numpy())

    return np.stack(losses), np.stack(yhats), h_hist


def plot_rnn_training(
    losses, yhats, x, start=0, colors=None, n_iter=5, session=0, col_buffer=3
):
    skip = len(losses) // n_iter
    colors = styles.default(colors)

    fig, ax = plt.subplots(1, x.shape[-1] + 1, figsize=(2 * (x.shape[-1] + 1), 2))
    ax[0].plot(losses, color=colors.subtle)

    buf = col_buffer * skip
    pal = colors.ch0(np.arange(start - buf, len(losses) + buf, skip))
    for i in range(x.shape[-1]):
        ax[i + 1].plot(x.numpy()[session, :, i], color=colors.subtle)
        for j in range(start, len(losses), skip):
            ax[0].plot([j], [losses[j]], "o", ms=3, color=pal[j])
            ax[i + 1].plot(yhats[j, session, :, i], color=pal[j])

    return fig, ax
