import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


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
    from dynrn.rnntasks import (
        DriscollTasks,
        DriscollPlots,
        period_start_mask,
        periwindows,
        periperiod_sliced,
        split_trials,
        extract_trial_data,
        apply_to_trial_groups,
        nanmean_with_req
    )
    from dynrn.predictors import activity_dataset, save_dsn, load_dsn, discounted_sums
    import dynrn.basic_rnns as rnns
    from dynrn.basic_rnns import find_hash
    from dynrn.driscoll_rnns import lowrank_space, lowrank_step, lowrank_trajectories
    from dynrn.viz import dynamics as vd
    import scipy.stats
    from cmap import Colormap
    from scipy.stats import uniform, norm
    from datetime import datetime
    from mplutil import util as vu
    import matplotlib.pyplot as plt
    from pathlib import Path
    from sklearn.decomposition import PCA
    import tqdm
    from numpy import linalg as la
    import os
    import joblib as jl
    import time
    import seaborn as sns
    return (
        Colormap,
        DriscollPlots,
        DriscollTasks,
        PCA,
        Path,
        activity_dataset,
        apply_to_trial_groups,
        copy,
        datetime,
        dill,
        discounted_sums,
        extract_trial_data,
        find_hash,
        jit,
        jl,
        la,
        load_dsn,
        lowrank_space,
        lowrank_step,
        lowrank_trajectories,
        namedtuple,
        nanmean_with_req,
        nn,
        norm,
        np,
        optim,
        os,
        period_start_mask,
        periperiod_sliced,
        periwindows,
        plt,
        rnns,
        save_dsn,
        scipy,
        sns,
        split_trials,
        th,
        time,
        tqdm,
        uniform,
        vd,
        vu,
    )


@app.cell
def __(Path, __file__):
    from dynrn.viz import styles
    from dynrn.viz.styles import getc
    t20 = lambda x: getc(f"seaborn:tab20{x}")
    colors, plotter = styles.init_plt(
        (Path(__file__).parent / '../plots/notebook/lorank-multi-bhv').resolve(),
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
    mo.md(r"""### Load task and network""")
    return


@app.cell
def __(Path, device, dill, find_hash, rnns):
    root_dir = Path("/Users/kaifox/projects/loop/dynrn/data")

    rnn_hash = '7c399f'
    rnn_path = find_hash(root_dir, rnn_hash, ext='.pt')
    rnn_name = Path(rnn_path).name
    if rnn_name.endswith(".pt"):
        rnn_name = rnn_name[:-3]
    rnn, ckpts, traindata = rnns.load_rnn(rnn_path, device=device)

    dset_hash = traindata['dataset_hash']
    dset = dill.load(open(find_hash(root_dir, dset_hash), 'rb'))['test']
    return (
        ckpts,
        dset,
        dset_hash,
        rnn,
        rnn_hash,
        rnn_name,
        rnn_path,
        root_dir,
        traindata,
    )


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### Network intrinsic dynamics""")
    return


@app.cell
def __(plotter, rnn_name):
    (plotter.plot_dir / rnn_name).mkdir(exist_ok = True)
    return


@app.cell
def __(lowrank_space, rnn):
    low_space = lowrank_space(rnn)
    return low_space,


@app.cell
def __(low_space, lowrank_trajectories, rnn):
    base_traj = lowrank_trajectories(
        rnn, 100, 50, (-5, 5), space = low_space
    )
    return base_traj,


@app.cell
def __(base_traj, colors, plotter, plt, rnn_name, vd):
    def _plot():
        fig, ax = plt.subplots(2, 3, figsize = (6, 3), sharex = 'col', sharey = 'col')
        timecolors = colors.ch0(30)
        vd.trajecories(ax[0, 0], base_traj[:, :], timecolors, lw = 1)
        vd.trajecories(ax[0, 1], base_traj[:, 10:], timecolors, lw = 1)
        vd.trajecories(ax[0, 2], base_traj[:, 20:], timecolors, lw = 1)
        ax[1, 0].scatter(*base_traj.T, s = 1, alpha = 0.1, color = 'k')
        ax[1, 1].scatter(*base_traj[:, 10:].T, s = 0.5, alpha = 0.1, color = 'k')
        ax[1, 2].scatter(*base_traj[:, 20:].T, s = 0.5, alpha = 0.1, color = 'k')
        ax[0, 0].set_title("t = 0...50")
        ax[0, 1].set_title("t = 10...50")
        ax[0, 2].set_title("t = 20...50")
        for a in ax.ravel():
            a.set_aspect(1.)
        return fig, ax

    _f, _ = _plot(); plotter.finalize(_f, f"{rnn_name}/dynamics-base"); _f
    return


@app.cell
def __(base_traj, np, plotter, plt):
    def _plot():
        fig, ax = plt.subplots(figsize = (2, 1.5))
        v = np.linalg.norm(np.diff(base_traj, axis=1), axis = -1)
        t = np.arange(v.shape[1])
        ax.plot(t[20:], v[:, 20:].T, color = 'k', lw = 1)
        ax.set_ylabel("velocity")
        return fig, ax

    _f, _ = _plot(); plotter.finalize(_f, None); _f
    return


@app.cell
def __(base_traj, plotter, plt):
    def _plot():
        fig, ax = plt.subplots(figsize = (1.5, 1.5))
        x = base_traj[:, 20:, 1][:, :-1]
        y = base_traj[:, 20:, 1][:, 1:]
        ax.scatter(x.ravel(), y.ravel(), c = 'k', s = 2)
        ax.set_xlabel(r'$\hbar_2$(t)')
        ax.set_ylabel(r'$\hbar_2$(t + 1)')
        return fig, ax

    _f, _ = _plot(); plotter.finalize(_f, None); _f
    return


@app.cell
def __(low_space, lowrank_trajectories, rnn, th):
    iti_x = th.zeros(100, 5)
    iti_x[:, 4] = 1
    iti_traj = lowrank_trajectories(
        rnn, 100, 50, (-40, 40), space = low_space, x = iti_x
    )
    return iti_traj, iti_x


@app.cell
def __(colors, iti_traj, plotter, plt, rnn_name, vd):
    def _plot():
        fig, ax = plt.subplots(2, 3, figsize = (6, 3), sharex = 'col', sharey = 'col')
        timecolors = colors.ch0(30)
        vd.trajecories(ax[0, 0], iti_traj[:, :], timecolors, lw = 1)
        vd.trajecories(ax[0, 1], iti_traj[:, 10:], timecolors, lw = 1)
        vd.trajecories(ax[0, 2], iti_traj[:, 20:], timecolors, lw = 1)
        ax[1, 0].scatter(*iti_traj.T, s = 1, alpha = 0.1, color = 'k')
        ax[1, 1].scatter(*iti_traj[:, 10:].T, s = 0.5, alpha = 0.1, color = 'k')
        ax[1, 2].scatter(*iti_traj[:, 20:].T, s = 0.5, alpha = 0.1, color = 'k')
        ax[0, 0].set_title("t = 0...50")
        ax[0, 1].set_title("t = 10...50")
        ax[0, 2].set_title("t = 20...50")
        for a in ax.ravel():
            a.set_aspect(1.)
        return fig, ax

    _f, _ = _plot(); plotter.finalize(_f, f"{rnn_name}/dynamics-iti"); _f
    return


@app.cell
def __(low_space, lowrank_trajectories, rnn, th):
    context_x = th.zeros(100, 5)
    context_x[:, 0] = 1
    context_x[:, 4] = 1
    context_traj = lowrank_trajectories(
        rnn, 100, 50, (-40, 40), space = low_space, x = context_x
    )
    return context_traj, context_x


@app.cell
def __(colors, context_traj, plotter, plt, rnn_name, vd):
    def _plot():
        fig, ax = plt.subplots(2, 3, figsize = (6, 3), sharex = 'col', sharey = 'col')
        timecolors = colors.ch0(30)
        vd.trajecories(ax[0, 0], context_traj[:, :], timecolors, lw = 1)
        vd.trajecories(ax[0, 1], context_traj[:, 10:], timecolors, lw = 1)
        vd.trajecories(ax[0, 2], context_traj[:, 20:], timecolors, lw = 1)
        ax[1, 0].scatter(*context_traj.T, s = 1, alpha = 0.1, color = 'k')
        ax[1, 1].scatter(*context_traj[:, 10:].T, s = 0.5, alpha = 0.1, color = 'k')
        ax[1, 2].scatter(*context_traj[:, 20:].T, s = 0.5, alpha = 0.1, color = 'k')
        ax[0, 0].set_title("t = 0...50")
        ax[0, 1].set_title("t = 10...50")
        ax[0, 2].set_title("t = 20...50")
        for a in ax.ravel():
            a.set_aspect(1.)
        return fig, ax

    _f, _ = _plot(); plotter.finalize(_f, f"{rnn_name}/dynamics-context-mem"); _f
    return


@app.cell
def __(low_space, lowrank_trajectories, rnn, th):
    stim_x = th.zeros(100, 5)
    stim_x[:, 0] = 1
    stim_x[:, 2] = th.cos(th.linspace(0, th.pi / 4, 100))
    stim_x[:, 3] = th.sin(th.linspace(0, th.pi / 4, 100))
    stim_x[:, 4] = 1
    stim_traj = lowrank_trajectories(
        rnn, 100, 50, (-40, 40), space = low_space, x = stim_x
    )
    return stim_traj, stim_x


@app.cell
def __(Colormap, colors, plotter, plt, rnn_name, stim_traj, vd):
    def _plot():
        fig, ax = plt.subplots(2, 3, figsize = (6, 3), sharex = 'col', sharey = 'col')
        timecolors = colors.ch0(30)
        vd.trajecories(ax[0, 0], stim_traj[:, :], timecolors, lw = 1)
        vd.trajecories(ax[0, 1], stim_traj[:, 10:], timecolors, lw = 1)
        vd.trajecories(ax[0, 2], stim_traj[:, 20:], timecolors, lw = 1)
        for i in range(100):
            c = Colormap('cool')(i / 100)
            ax[1, 0].scatter(*stim_traj[i].T, s = 1, alpha = 0.1, color = c)
            ax[1, 1].scatter(*stim_traj[i, 10:].T, s = 0.5, alpha = 0.1, color = c)
            ax[1, 2].scatter(*stim_traj[i, 20:].T, s = 0.5, alpha = 0.1, color = c)
        ax[0, 0].set_title("t = 0...50")
        ax[0, 1].set_title("t = 10...50")
        ax[0, 2].set_title("t = 20...50")
        for a in ax.ravel():
            a.set_aspect(1.)
        return fig, ax

    _f, _ = _plot(); plotter.finalize(_f, f"{rnn_name}/dynamics-stim"); _f
    return


@app.cell
def __(low_space, lowrank_trajectories, rnn, th):
    resp_x = th.zeros(100, 5)
    resp_x[:, 1] = 1
    resp_x[:, 4] = 1
    resp_traj = lowrank_trajectories(
        rnn, 100, 50, (-40, 40), space = low_space, x = resp_x
    )
    return resp_traj, resp_x


@app.cell
def __(colors, plotter, plt, resp_traj, rnn_name, vd):
    def _plot():
        fig, ax = plt.subplots(2, 3, figsize = (6, 3), sharex = 'col', sharey = 'col')
        timecolors = colors.ch0(30)
        vd.trajecories(ax[0, 0], resp_traj[:, :], timecolors, lw = 1)
        vd.trajecories(ax[0, 1], resp_traj[:, 10:], timecolors, lw = 1)
        vd.trajecories(ax[0, 2], resp_traj[:, 20:], timecolors, lw = 1)
        ax[1, 0].scatter(*resp_traj.T, s = 1, alpha = 0.1, color = 'k')
        ax[1, 1].scatter(*resp_traj[:, 10:].T, s = 0.5, alpha = 0.1, color = 'k')
        ax[1, 2].scatter(*resp_traj[:, 20:].T, s = 0.5, alpha = 0.1, color = 'k')
        ax[0, 0].set_title("t = 0...50")
        ax[0, 1].set_title("t = 10...50")
        ax[0, 2].set_title("t = 20...50")
        for a in ax.ravel():
            a.set_aspect(1.)
        return fig, ax

    _f, _ = _plot(); plotter.finalize(_f, f"{rnn_name}/dynamics-resp"); _f
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
