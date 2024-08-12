import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


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
    )
    import dynrn.basic_rnns as rnns
    from dynrn.basic_rnns import timehash, find_hash
    import scipy.stats
    from scipy.stats import uniform, norm
    from datetime import datetime
    from scipy.spatial.distance import cosine as cosine_dist
    from mplutil import util as vu
    import scipy.linalg
    import matplotlib.pyplot as plt
    from pathlib import Path
    from sklearn.decomposition import PCA
    from cmap import Colormap
    import tqdm
    import os
    import joblib as jl
    import time
    import glob
    import seaborn as sns
    return (
        Colormap,
        DriscollTasks,
        PCA,
        Path,
        activity_dataset,
        copy,
        cosine_dist,
        create_memorypro_activity_dataset,
        datetime,
        dill,
        find_hash,
        fit_dsn,
        glob,
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
    colors, plotter = styles.init_plt(
        '../plots/notebook/multi-rnn-performance',
        fmt = 'pdf', display=False)
    plot_root = Path(plotter.plot_dir)
    return colors, getc, plot_root, plotter, styles


@app.cell
def __(th):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    cpu = th.device('cpu' if th.cuda.is_available() else 'cpu')
    print(device.type)
    return cpu, device


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Load rnn model and training data""")
    return


@app.cell
def __(Path, device, dill, find_hash, rnns):
    root_dir = Path("/Users/kaifox/projects/loop/dynrn/data")

    rnn_hash = "7b7844"
    rnn_path = find_hash(root_dir, rnn_hash, ".pt")
    final_rnn, rnn_ckpts, traindata = rnns.load_rnn(rnn_path, device=device)

    task_data_hash = traindata['dataset_hash']
    task_data = dill.load(open(find_hash(root_dir, task_data_hash, '.dil'), 'rb'))
    task_meta_hash = traindata['task_hash']
    task_meta = dill.load(open(find_hash(root_dir, task_meta_hash, '.dil'), 'rb'))
    return (
        final_rnn,
        rnn_ckpts,
        rnn_hash,
        rnn_path,
        root_dir,
        task_data,
        task_data_hash,
        task_meta,
        task_meta_hash,
        traindata,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Evaluate validation losses""")
    return


@app.cell
def __(device, final_rnn, rnn_ckpts, task_data, th, tqdm):
    test_data = task_data["test"]
    n_ex_session = 1

    test_preds = {}
    test_losses = {}

    x = th.tensor(test_data["stimuli"], dtype=th.float32).to(device)
    h_init = th.zeros([x.shape[0], final_rnn.nh])
    for step, rnn in tqdm.tqdm(rnn_ckpts.items()):
        all_preds, all_h = rnn.seq_forward(x, h_init)
        test_losses[step] = (all_preds.detach() - test_data["targets"]) ** 2
        test_preds[step] = [
            all_h[slc][:n_ex_session].detach().numpy()
            for slc in test_data['block_slices']
        ]
        # if step >= 1000:
        #     break
    return (
        all_h,
        all_preds,
        h_init,
        n_ex_session,
        rnn,
        step,
        test_data,
        test_losses,
        test_preds,
        x,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### Plot""")
    return


@app.cell
def __(
    colors,
    np,
    plotter,
    plt,
    rnns,
    test_data,
    test_losses,
    test_preds,
    traindata,
):
    block = 0
    session = 0
    n_ex_steps = 3

    _tgt = test_data['targets'][test_data['block_slices'][block]][[session]]
    _steps = np.array(sorted(list(test_losses.keys())))
    _ex_steps = _steps[np.linspace(0, len(_steps) - 1, n_ex_steps).astype('int')]
    _yhats = {i: test_preds[i][block][[session]] for i in _steps}
    _losses = {i: test_losses[i].mean() for i in _steps}

    ncol = test_data['n_tgt'] + 1

    fig, ax = plt.subplots(1, ncol, figsize = (2 * ncol, 1.5))
    rnns.plot_rnn_training(
        _losses,
        _yhats,
        _tgt,
        epochs = _steps,
        ex_epochs = _ex_steps,
        ax = ax,
        loss_matches_epochs=True
    )
    ax[0].set_yscale('log')
    ax[0].plot(traindata['losses'], color = colors.neutral, lw = 0.5, zorder = -1)
    plotter.finalize(fig, None)
    fig
    return ax, block, fig, n_ex_steps, ncol, session


@app.cell
def __(Colormap, np, plotter, plt, test_data, test_losses):
    n_blocks = len(test_data['block_slices'])
    task_colors = Colormap('crest')(np.linspace(0.2, 0.8, n_blocks))
    _steps = np.array(sorted(list(test_losses.keys())))

    _fig, _ax = plt.subplots(1,1, figsize=(2.5,1.6))
    for _i_bl in range(n_blocks):
        block_losses = {i: test_losses[i][test_data['block_slices'][_i_bl]].mean() for i in _steps}
        _ax.plot(_steps, [block_losses[i] for i in _steps], color = task_colors[_i_bl])
    _ax.set_yscale('log')
    _ax.set_xlabel("Training step")
    _ax.set_ylabel("Block loss")
    plotter.finalize(_fig, None)
    _fig
    return block_losses, n_blocks, task_colors


@app.cell
def __(
    DriscollTasks,
    block,
    np,
    plt,
    session,
    test_data,
    test_losses,
    test_preds,
):
    block_data = DriscollTasks.split_dataset(test_data)
    _task = block_data[block]["task"]
    _steps = np.array(sorted(list(test_losses.keys())))
    _best_step = sorted(_steps, key=lambda i: test_losses[i].mean())[0]
    print("best step:", _best_step)
    _nax = len(_task.stim_groups) + 2 * len(_task.tgt_groups)

    _fig, _ax = plt.subplots(_nax, 1, figsize=(4, _nax * 0.75), sharex=True)
    print(-len(_task.tgt_groups), len(_task.stim_groups))
    print(_ax[-len(_task.tgt_groups) :], _ax[len(_task.stim_groups) :])
    for _a1, _a2 in zip(
        _ax[-len(_task.tgt_groups) :], _ax[len(_task.stim_groups) :]
    ):
        _a2.sharey(_a1)
        _a1.set_ylabel("rnn outputs")
        _a2.set_ylabel("targets")
    DriscollTasks.plot_session(
        block_data[block], session=session, ax=_ax, legend=True
    )
    _pred = test_preds[_best_step][block][[session]]
    # arrange axes for plot_session to only plot outputs (which it thinks are targets)
    _pred_ax = [None] * len(_task.stim_groups) + _ax[
        -len(_task.tgt_groups) :
    ].tolist()
    DriscollTasks.plot_session(_task, y=_pred, session=0, ax=_pred_ax)
    _fig
    return block_data,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
