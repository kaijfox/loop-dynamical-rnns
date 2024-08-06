import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### Setup and import""")
    return


@app.cell
def __():
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
        plot_memorypro_prediction,
        plot_memorypro_dsn_prediction
    )
    import dynrn.basic_rnns as rnns
    from dynrn.basic_rnns import timehash, find_hash
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
    return (
        DriscollTasks,
        PCA,
        Path,
        activity_dataset,
        copy,
        create_memorypro_activity_dataset,
        datetime,
        dill,
        find_hash,
        fit_dsn,
        itiexp,
        jit,
        jl,
        load_dsn,
        namedtuple,
        nn,
        norm,
        np,
        optim,
        os,
        plot_memorypro_dsn_prediction,
        plot_memorypro_prediction,
        plt,
        rnns,
        save_dsn,
        scipy,
        sns,
        td_loss,
        th,
        time,
        timehash,
        tqdm,
        uniform,
        vu,
    )


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(Path):
    from dynrn.viz import styles
    from dynrn.viz.styles import getc
    t20 = lambda x: getc(f"seaborn:tab20{x}")
    colors, plotter = styles.init_plt(
        '../plots/notebook/mlp-predict',
        fmt = 'pdf', display=False)
    plot_root = Path(plotter.plot_dir)
    return colors, getc, plot_root, plotter, styles, t20


@app.cell
def __(th):
    # cuda setup
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    cpu = th.device('cpu' if th.cuda.is_available() else 'cpu')
    print(device.type)
    return cpu, device


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Load RNN and get hidden trajectories

        (from Driscoll task)
        """
    )
    return


@app.cell
def __(Path, device, jit, jl, th):
    # root_dir = Path("/n/home08/kfox/sabalab/kfox/loop/dynrn")
    root_dir = Path("/Users/kaifox/projects/loop/dynrn/data")

    model_name = "driscoll-n1024_5c135.07181253"
    model_hash = model_name.split('_')[-1]
    model_path = root_dir / "task-models" / model_name
    model = jit.load(f"{model_path}.pt", map_location=device)
    model.load_state_dict(th.load(f"{model_path}.tar", map_location=device))

    task_kws = jl.load(root_dir / 'tasks' / "memorypro-long.jl")

    dataset_kws = dict(
        n_dim = -1,
        n_train = 200,
        n_val = 50,
        seed_train = 1,
        seed_val = 2,
        session_length = 500,
        task_kws = task_kws
    )
    dset_hash = '7add88' #timehash()
    dataset_path = root_dir / 'pred-datasets' / f"mempro-r.none-m{model_hash}_{dset_hash}.dil"
    return (
        dataset_kws,
        dataset_path,
        dset_hash,
        model,
        model_hash,
        model_name,
        model_path,
        root_dir,
        task_kws,
    )


@app.cell
def __(
    create_memorypro_activity_dataset,
    dataset_kws,
    dataset_path,
    dill,
    jl,
    model,
    model_name,
):
    # Load dataset if already exists
    if dataset_path.exists():
        print("Loaded dataset:", str(dataset_path))
        data = jl.load(dataset_path)['dataset']

    # Otherwise create and save
    else:
        if not dataset_path.parent.exists():
            dataset_path.parent.mkdir(parents=True)
        
        _metadata = {
            'data_source': 'dynrn.predictors.create_memorypro_activity_dataset',
            'rnn_model': model_name,
            'source_kws': dataset_kws
        }
            
        data = create_memorypro_activity_dataset(
            model, **dataset_kws
        )
        dill.dump({'dataset': data, 'metadata': _metadata}, open(dataset_path, 'wb'))
        print("Created dataset:", str(dataset_path))
    return data,


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### Training""")
    return


@app.cell
def __(find_hash, root_dir):
    dsn_hash = '7add5a.dil' #timehash()
    # dsn_path = root_dir / "pred-models" / f"td-g0.9-b20_m.{model_hash}-d.{dset_hash}_{dsn_hash}"
    dsn_path = find_hash(root_dir, dsn_hash, ext='.pt')
    return dsn_hash, dsn_path


@app.cell
def __(Path, device, dsn_path, load_dsn):
    # Load model if it has already been saved
    if not Path(dsn_path).exists():
        print(f"No such model: {dsn_path}")
    else:
        mlp, ckpts, _train_data = load_dsn(dsn_path, device)
        cumulant_fn = _train_data['cumulant_fn']
        losses = _train_data['losses']
        gamma = _train_data['gamma']
        print(f"Loaded DSN model: {dsn_path}")
    return ckpts, cumulant_fn, gamma, losses, mlp


@app.cell(disabled=True)
def __(data, device, dsn_path, fit_dsn, jit, nn, optim, save_dsn, th):
    # Train DSN model and save to `dsn_path`
    # This cell defines only local variables so that it can be deactivated

    _widths = [512 // i for i in [2, 4]]
    _reduction = 20
    _layers = lambda: (
        nn.Linear(data.train.n_act, _reduction),
        nn.Linear(_reduction, _widths[0]),
        nn.Softplus(),
        nn.LayerNorm(_widths[0]),
        nn.Linear(_widths[0], _widths[1]),
        nn.Softplus(),
        nn.Linear(_widths[1], data.train.n_act)
    )

    _x = data.train.activity
    _y = data.train.activity[:, 1:]

    th.manual_seed(0)
    _mlp = jit.script(nn.Sequential(*_layers()))
    _mlp.to(device)
    _opt = optim.Adam(_mlp.parameters(), lr=1e-3)
    _gamma = 0.9
    _cumulant_fn = lambda d: d.activity[:, 1:]

    _losses, _ckpts = fit_dsn(
        _mlp,
        _x.to(device),
        _y.to(device),
        _opt,
        gamma=_gamma,
        n_steps=601,
        loss_fn=nn.MSELoss(),
        checkpoint_every=100,
    )
    save_dsn(
        dsn_path,
        _mlp,
        data,
        gamma=_gamma,
        cumulant_fn=_cumulant_fn,
        checkpoints=_ckpts,
        losses=_losses,
    )
    print(f"Saved DSN model: {dsn_path}")
    return


@app.cell
def __(losses, plotter, plt):
    fig, ax = plt.subplots(1, 1, figsize = (1.2, 1))
    ax.plot(losses)
    plotter.finalize(fig, None, display = False)
    fig
    return ax, fig


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Training results""")
    return


@app.cell
def __(
    activity_dataset,
    colors,
    discounted_sums,
    getc,
    nn,
    np,
    plt,
    rnns,
    t20,
    td_loss,
    th,
):
    def plot_driscoll_ax(ax, x, y, periods, t=None, session=0, periods_only=False):
        icolors = [getc("k"), getc("grey"), t20("b:17"), t20("b:19")]
        ocolors = [t20("c:0"), t20("c:2")]
        if t is None:
            t = np.arange(x.shape[1])

        if not periods_only:
            ax.plot(t, x.numpy()[session, :, 2], color=icolors[2], lw=1)
            ax.plot(t, x.numpy()[session, :, 3], color=icolors[3], lw=1)
            ax.plot(t, y.numpy()[session, :, 0], color=ocolors[0], lw=1)
            ax.plot(t, y.numpy()[session, :, 1], color=ocolors[1], lw=1)
        for i in np.where(np.diff(periods[session]) != 0)[0]:
            ax.axvline(
                t[i],
                color=".6" if periods[session, i] == 0 else ".9",
                lw=1,
                zorder=-1,
            )


    def _plot_memorypro_prediction(
        losses,
        dataset,
        predictions,
        targets,
        losses_val,
        plot_units,
        session=0,
    ):
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
        rnns.plot_rnn_training(
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
                plot_driscoll_ax(
                    ax[r, i + 1],
                    dataset.stim,
                    dataset.targets,
                    dataset.periods,
                    session=session,
                    periods_only=not r,
                )
                if r == 1:
                    ax[r, i + 1].twinx().plot(
                        dataset.activity[session, :, plot_units[i]],
                        color=colors.neutral,
                        lw=1,
                    )

        return fig, ax


    def _plot_memorypro_dsn_prediction(
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
                i: checkpoints[i].cpu()(val_dataset.activity.cpu()).detach()
                for i in steps
            }
        ckpt_preds = {i: y[..., plot_units].numpy() for i, y in ckpt_yhats.items()}
        ckpt_loss = {
            i: td_loss(y, cumulant, gamma, loss_fn) for i, y in ckpt_yhats.items()
        }
        manual_sums = discounted_sums(
            cumulant[..., plot_units].transpose(2, 1), gamma
        ).transpose(2, 1)

        fig, ax = _plot_memorypro_prediction(
            losses,
            val_dataset,
            ckpt_preds,
            manual_sums,
            ckpt_loss,
            plot_units,
            session,
        )
        ax[0, 0].set_yscale("log")

        return fig, ax
    return plot_driscoll_ax,


@app.cell
def __(ckpts, data, gamma, losses, plot_memorypro_dsn_prediction, plotter):
    _y_val = data.val.activity[:, 1:]
    _fig, _ax = plot_memorypro_dsn_prediction(
        losses, data.val, _y_val, gamma, ckpts, [0, 1]
    )
    plotter.finalize(_fig, None, display=False)
    _fig
    return


@app.cell
def __(np, th):
    def discounted_sums(x, gamma):
        """
        Parameters
        ----------
        x : array, shape (.., n_samples)
        gamma : float
        """
        if th.is_tensor(x): m = th
        else: m = np
        T = x.shape[-1]
        gamma = gamma ** m.arange(T)
        return m.stack([
            (x[..., i:] * gamma[:T-i]).sum(axis = -1)
            for i in range(T)
        ], axis = -1)
    return discounted_sums,


@app.cell
def __(data, discounted_sums, plotter, plt):
    _fig, _ax = plt.subplots(1, 1, figsize = (5, 2))
    _c = data.val.activity[:, 1:, 0]
    _sums = discounted_sums(_c, 0.9)
    print(_c.shape)
    _ax.plot(_c[0])
    _ax.plot(_sums[0])
    plotter.finalize(_fig, None, display = False)
    _fig
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
