import marimo

__generated_with = "0.7.12"
app = marimo.App()


app._unparsable_cell(
    r"""
    %load_ext autoreload
    %autoreload 3 --print
    """,
    name="__"
)


@app.cell
def __():
    import torch as th
    import torch.jit as jit
    import copy
    import numpy as np
    from torch import optim
    from torch import nn
    from dynrn.rnntasks import DriscollTasks, itiexp
    import dynrn.basic_rnns as rnns
    import scipy.stats
    from scipy.stats import uniform, norm
    from datetime import datetime
    from mplutil import util as vu
    import matplotlib.pyplot as plt
    from pathlib import Path
    import tqdm
    import joblib as jl
    import time
    import seaborn as sns
    return (
        DriscollTasks,
        Path,
        copy,
        datetime,
        itiexp,
        jit,
        jl,
        nn,
        norm,
        np,
        optim,
        plt,
        rnns,
        scipy,
        sns,
        th,
        time,
        tqdm,
        uniform,
        vu,
    )


app._unparsable_cell(
    r"""
    %matplotlib inline
    %config InlineBackend.figure_format='retina'
    from dynrn.viz import styles
    from dynrn.viz.styles import getc
    t20 = lambda x: getc(f\"seaborn:tab20{x}\")
    colors, plotter = styles.init_plt(
        '../plots/notebook/mlp-predict',
        fmt = 'pdf')
    plot_root = Path(plotter.plot_dir)
    """,
    name="__"
)


@app.cell
def __(th):
    # cuda setup
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    cpu = th.device('cpu' if th.cuda.is_available() else 'cpu')
    print(device.type)
    return cpu, device


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Load RNN and get hidden trajectories
        """
    )
    return


@app.cell
def __(Path, device, jit, jl, th):
    # model_root = Path("/n/home08/kfox/sabalab/kfox/loop/dynrn/task-models")
    model_root = Path("/Users/kaifox/projects/loop/dynrn/data/task-models")
    # model_path = model_root / "driscoll-long-alpha1-n1024-test_5c0ae.07181038"
    model_path = model_root / "driscoll-n1024_5c135.07181253"
    train = jl.load(f"{model_path}.train.jl")
    model = jit.load(f"{model_path}.pt", map_location=device)
    model.load_state_dict(th.load(f"{model_path}.tar", map_location=device))

    task_kws = jl.load(model_root / '../tasks' / "memorypro-long.jl")
    return model, model_path, model_root, task_kws, train


@app.cell
def __(DriscollTasks, task_kws, th):
    x, y, periods = DriscollTasks.memorypro(
        session_length=500,
        n_sessions=200,
        seed = 1,
        **task_kws
    )
    x = th.tensor(x, dtype=th.float32)
    y = th.tensor(y, dtype=th.float32)

    xval, yval, periodsval = DriscollTasks.memorypro(
        session_length=500,
        n_sessions=50,
        seed = 2,
        **task_kws
    )
    xv = th.tensor(xval, dtype=th.float32)
    yv = th.tensor(yval, dtype=th.float32)
    return periods, periodsval, x, xv, xval, y, yv, yval


@app.cell
def __(cpu, model, th, x, xv):
    h_init = th.zeros(x.shape[0], model.nh)
    yhat, h = model.to(cpu).seq_forward(x, h_init)

    h_init = th.zeros(xv.shape[0], model.nh)
    yhatv, hv = model.to(cpu).seq_forward(xv, h_init)
    return h, h_init, hv, yhat, yhatv


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Train next-step predictor
        """
    )
    return


@app.cell
def __(h):
    tofs = 1

    xh = h[:, :-tofs].detach()
    yh = h[:, tofs:].detach()

    return tofs, xh, yh


@app.cell
def __(hv, tofs):
    xhv = hv[:, :-tofs].detach()
    yhv = hv[:, tofs:].detach()
    return xhv, yhv


@app.cell
def __(device, nn, optim, rnns, xh, yh):
    mlp = nn.Sequential(
        nn.Linear(xh.shape[-1], xh.shape[-1] // 2),
        nn.Softplus(),
        nn.LayerNorm(xh.shape[-1] // 2),
        nn.Linear(xh.shape[-1] // 2, xh.shape[-1] // 4),
        nn.Softplus(),
        nn.Linear(xh.shape[-1] // 4, yh.shape[-1])
    ).to(device)
    opt = optim.Adam(mlp.parameters(), lr=1e-3)

    losses, yhats, ckpts = rnns.fit_ffn(
        mlp,
        xh.to(device),
        yh.to(device),
        opt,
        n_steps=5001,
        return_batches=None,
        checkpoint_every = 1000,
    )
    return ckpts, losses, mlp, opt, yhats


@app.cell
def __(Path, ckpts, jl, losses, yhats):
    out_dir = Path(f"/n/home08/kfox/sabalab/kfox/loop/dynrn/predictors")
    jl.dump({
        'ckpts': ckpts, 'losses': losses, 'yhats': yhats
    }, out_dir / "ffn-memorypro-tmp.jl")
    return out_dir,


@app.cell
def __(ckpts, nn, xh, xhv, yh, yhv):
    plot_units = [0, 1]
    # evaluate each model checkpoint and pull a few units
    ckpt_yhats =  [m(xh).detach().numpy() for m in ckpts]
    yhat_selected = sum([[yhat[..., plot_units]] + [None] * 999 for yhat in ckpt_yhats], [])
    yh_selected = yh[..., plot_units]

    yhatsv = [m(xhv).detach().cpu() for m in ckpts]
    lossesv = [nn.MSELoss()(yhat, yhv) for yhat in yhatsv]
    return (
        ckpt_yhats,
        lossesv,
        plot_units,
        yh_selected,
        yhat_selected,
        yhatsv,
    )


@app.cell
def __(
    colors,
    losses,
    lossesv,
    np,
    plotter,
    rnns,
    yh_selected,
    yhat_selected,
):
    ts = np.arange(0, 5001, 1000)

    fig, ax = rnns.plot_rnn_training(losses, yhat_selected, yh_selected, epochs = ts)
    ax[0].set_yscale('log')
    ax[0].plot(ts, lossesv, color = colors.C[0])
    plotter.finalize(fig, None)
    return ax, fig, ts


@app.cell
def __(colors, periods, plot_driscoll_ax, plt, rnns, x, xh, y):
    def plot_memorypro_prediction(
        losses, yhat_selected, yh_selected, losses_val, epochs, plot_units, session=0
    ):
        n_units = len(plot_units)
        fig, ax = plt.subplots(
            2,
            n_units + 1,
            figsize=(2 + 3 * n_units, 2),
            sharex="col",
            width_ratios=[1] + [2] * n_units,
        )
        rnns.plot_rnn_training(
            losses,
            yhat_selected,
            yh_selected,
            epochs=epochs,
            session=session,
            ax = ax[0, :]
        )

        ax[0, 0].set_yscale("log")
        ax[1, 0].set_axis_off()
        ax[0, 0].plot(epochs, losses_val, color = colors.neutral, zorder = -1)

        for r in range(2):
            for i in range(0, n_units):
                plot_driscoll_ax(
                    ax[r, i + 1],
                    x,
                    y,
                    periods,
                    session=session,
                    periods_only=not r,
                )
                if r == 1:
                    ax[r, i + 1].twinx().plot(
                        xh[session, :, plot_units[i]],
                        color=colors.neutral,
                        lw=1,
                    )

        return fig, ax
    return plot_memorypro_prediction,


@app.cell
def __(
    losses,
    lossesv,
    plot_memorypro_prediction,
    plot_units,
    plotter,
    ts,
    yh_selected,
    yhat_selected,
):

    fig, ax = plot_memorypro_prediction(losses, yhat_selected, yh_selected, lossesv, ts, plot_units)

    plotter.finalize(fig, None)
    return ax, fig


@app.cell
def __(mo):
    mo.md(
        r"""
        ##### without batchnorm
        Hard to deal with in sign-constained models. Does this training work without it?

        Spoiler: it does.
        """
    )
    return


@app.cell
def __(device, nn, optim, rnns, xh, yh):
    mlp = nn.Sequential(
        nn.Linear(xh.shape[-1], xh.shape[-1] // 2),
        nn.Softplus(),
        nn.Linear(xh.shape[-1] // 2, xh.shape[-1] // 4),
        nn.Softplus(),
        nn.Linear(xh.shape[-1] // 4, yh.shape[-1])
    ).to(device)
    opt = optim.Adam(mlp.parameters(), lr=1e-3)

    losses, yhats, ckpts = rnns.fit_ffn(
        mlp,
        xh.to(device),
        yh.to(device),
        opt,
        n_steps=5001,
        return_batches=None,
        checkpoint_every = 1000,
    )
    return ckpts, losses, mlp, opt, yhats


@app.cell
def __(mo):
    mo.md(
        r"""

        """
    )
    return


@app.cell
def __(ckpts, losses, plotter, rnns, xh, yh):
    plot_units = [0, 1, 2]

    # evaluate each model checkpoint and pull a few units
    ckpt_yhats =  [m(xh).detach().numpy() for m in ckpts]
    yhat_selected = sum([[yhat[..., plot_units]] + [None] * 999 for yhat in ckpt_yhats], [])
    yh_selected = yh[..., plot_units]

    fig, ax = rnns.plot_rnn_training(losses, yhat_selected, yh_selected, epochs = range(0, 5001, 1000))
    ax[0].set_yscale('log')
    plotter.finalize(fig, None)
    return ax, ckpt_yhats, fig, plot_units, yh_selected, yhat_selected


@app.cell
def __(mo):
    mo.md(
        r"""
        ##### longer $t_{\text{ofs}}$
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        $t_{\text{ofs}} = 10$ 
        """
    )
    return


@app.cell
def __(h):

    tofs = 10
    xh = h[:, :-tofs].detach()
    yh = h[:, tofs:].detach()

    return tofs, xh, yh


@app.cell
def __(device, nn, optim, rnns, xh, yh):
    mlp = nn.Sequential(
        nn.Linear(xh.shape[-1], xh.shape[-1] // 2),
        nn.Softplus(),
        nn.Linear(xh.shape[-1] // 2, xh.shape[-1] // 4),
        nn.Softplus(),
        nn.Linear(xh.shape[-1] // 4, yh.shape[-1])
    ).to(device)
    opt = optim.Adam(mlp.parameters(), lr=1e-3)

    losses, yhats, ckpts = rnns.fit_ffn(
        mlp,
        xh.to(device),
        yh.to(device),
        opt,
        n_steps=5001,
        return_batches=None,
        checkpoint_every = 1000,
    )
    return ckpts, losses, mlp, opt, yhats


@app.cell
def __(getc, np, t20):
    def plot_driscoll_ax(ax, x, y, periods, t = None, session = 0, periods_only = False):
        icolors = [getc("k"), getc("grey"), t20("b:17"), t20("b:19")]
        ocolors = [t20("c:0"), t20("c:2")]
        if t is None:
            t = np.arange(x.shape[1])

        if not periods_only:
            # ax.plot(t, x.numpy()[session, :, 0], color = icolors[0], lw = 0.5, alpha = 0.3)
            # ax.plot(t, x.numpy()[session, :, 1], color = icolors[1], lw = 0.5, alpha = 0.3)
            ax.plot(t, x.numpy()[session, :, 2], color = icolors[2], lw = 1)
            ax.plot(t, x.numpy()[session, :, 3], color = icolors[3], lw = 1)
            ax.plot(t, y.numpy()[session, :, 0], color = ocolors[0], lw = 1)
            ax.plot(t, y.numpy()[session, :, 1], color = ocolors[1], lw = 1)
        for i in np.where(np.diff(periods[session]) != 0)[0]:
            ax.axvline(t[i], color='.6' if periods[session, i] == 0 else '.9', lw = 1, zorder = -1)

    return plot_driscoll_ax,


@app.cell
def __(
    ckpts,
    colors,
    losses,
    np,
    periods,
    plot_driscoll_ax,
    plotter,
    plt,
    rnns,
    x,
    xh,
    y,
    yh,
):
    plot_units = [0, 1]

    fig, ax = plt.subplots(2, len(plot_units) + 1, figsize=(12, 2), sharex="col", width_ratios = [1] + [2] * len(plot_units))

    # evaluate each model checkpoint and pull a few units
    ckpt_yhats = [m(xh).detach().numpy() for m in ckpts]
    yhat_selected = sum([[yhat[..., plot_units]] + [None] * 999 for yhat in ckpt_yhats], [])
    yh_selected = yh[..., plot_units]

    rnns.plot_rnn_training(
        losses, yhat_selected, yh_selected, epochs=range(0, 5001, 1000), ax=ax[0]
    )
    ax[0, 0].set_yscale("log")
    ax[1, 0].set_axis_off()
    for r in range(2):
        for i in range(0, len(plot_units)):
            plot_driscoll_ax(
                ax[r, i + 1],
                x,
                y,
                periods,
                t=np.arange(0, y.shape[1]),
                session=0,
                periods_only=not r,
            )
            if r == 1:
                ax[r, i + 1].twinx().plot(
                    xh[0, :, plot_units[i]],
                    color = colors.neutral,
                    lw = 1,
                )



    plotter.finalize(fig, None)
    return ax, ckpt_yhats, fig, i, plot_units, r, yh_selected, yhat_selected


@app.cell
def __(mo):
    mo.md(
        r"""
        $t_{\text{ofs}} = 50$ 
        """
    )
    return


@app.cell
def __(h):

    tofs = 50
    xh = h[:, :-tofs].detach()
    yh = h[:, tofs:].detach()

    return tofs, xh, yh


@app.cell
def __(device, nn, optim, rnns, xh, yh):
    mlp = nn.Sequential(
        nn.Linear(xh.shape[-1], xh.shape[-1] // 2),
        nn.Softplus(),
        nn.Linear(xh.shape[-1] // 2, xh.shape[-1] // 4),
        nn.Softplus(),
        nn.Linear(xh.shape[-1] // 4, yh.shape[-1])
    ).to(device)
    opt = optim.Adam(mlp.parameters(), lr=1e-3)

    losses, yhats, ckpts = rnns.fit_ffn(
        mlp,
        xh.to(device),
        yh.to(device),
        opt,
        n_steps=5001,
        return_batches=None,
        checkpoint_every = 1000,
    )
    return ckpts, losses, mlp, opt, yhats


@app.cell
def __(
    ckpts,
    colors,
    losses,
    np,
    periods,
    plot_driscoll_ax,
    plotter,
    plt,
    rnns,
    x,
    xh,
    y,
    yh,
):
    plot_units = [0, 1]

    fig, ax = plt.subplots(2, len(plot_units) + 1, figsize=(12, 2), sharex="col", width_ratios = [1] + [2] * len(plot_units))

    # evaluate each model checkpoint and pull a few units
    ckpt_yhats = [m(xh).detach().numpy() for m in ckpts]
    yhat_selected = sum([[yhat[..., plot_units]] + [None] * 999 for yhat in ckpt_yhats], [])
    yh_selected = yh[..., plot_units]

    rnns.plot_rnn_training(
        losses, yhat_selected, yh_selected, epochs=range(0, 5001, 1000), ax=ax[0]
    )
    ax[0, 0].set_yscale("log")
    ax[1, 0].set_axis_off()
    for r in range(2):
        for i in range(0, len(plot_units)):
            plot_driscoll_ax(
                ax[r, i + 1],
                x,
                y,
                periods,
                t=np.arange(0, y.shape[1]),
                session=0,
                periods_only=not r,
            )
            if r == 1:
                ax[r, i + 1].twinx().plot(
                    xh[0, :, plot_units[i]],
                    color = colors.neutral,
                    lw = 1,
                )



    plotter.finalize(fig, None)
    return ax, ckpt_yhats, fig, i, plot_units, r, yh_selected, yhat_selected


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### successor representation

        TD update to learn discounted sum
        """
    )
    return


@app.cell
def __(copy, nn, np, tqdm, yhats):
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

        For a black box differentiable function F(x) and sequence data
        $x_t\in\mathbb{R}^n$, $y_t\in\mathbb{R}^m$, minimize the objective
        $L = ||y_{t->t+1} + gamma * F(x_{t+1}) - F(x_t)||^2$
        if `loss_fn` is nn.MSELoss(), or more generally minimize
        $L = loss_fn(y_{t+1} + gamma * F(x_{t+1}), F(x_t)).$

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
        yhats : list[np.ndarray]
            The predicted output values at each epoch. These are not stacked because
            they can be large.
        models : List(nn.Module)
            The model at each checkpoint. Only returned if `checkpoint_every` is not None.
        """

        losses = []
        models = []

        for i in tqdm.trange(n_steps):
            opt.zero_grad()
            Fx = nn(x)
            Fx_tplus1 = Fx[:, 1:]
            Fx_t = Fx[:, :-1]
            loss = loss_fn(y + Fx_tplus1, Fx_t)
            loss.backward()
            opt.step()

            losses.append(loss.detach().cpu().numpy())
            if checkpoint_every is not None and i % checkpoint_every == 0:
                models.append(copy.deepcopy(nn).cpu())

        ret = (np.array(losses), yhats)
        if checkpoint_every is not None:
            ret += (models,)
        return ret

    return fit_dsn,


@app.cell
def __(device, jit, nn, optim, rnns, xh, yh):
    mlp = nn.Sequential(
        nn.Linear(xh.shape[-1], xh.shape[-1] // 2),
        nn.Softplus(),
        nn.LayerNorm(xh.shape[-1] // 2),
        nn.Linear(xh.shape[-1] // 2, xh.shape[-1] // 4),
        nn.Softplus(),
        nn.Linear(xh.shape[-1] // 4, yh.shape[-1])
    )
    mlp_script = jit.script(mlp)
    mlp_script.to(device)
    opt = optim.Adam(mlp.parameters(), lr=1e-3)

    losses, yhats, ckpts = rnns.fit_ffn(
        mlp_script,
        xh.to(device),
        yh.to(device),
        opt,
        n_steps=51,
        return_batches=None,
        checkpoint_every = 1000,
    )
    return ckpts, losses, mlp, mlp_script, opt, yhats


@app.cell
def __():
    return


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()

