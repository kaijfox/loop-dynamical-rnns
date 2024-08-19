import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md("""### Setup""")
    return


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
        nanmean_with_req,
        split_trials_driscoll,
    )
    from dynrn.predictors import activity_dataset, save_dsn, load_dsn, discounted_sums
    import dynrn.basic_rnns as rnns
    from dynrn.basic_rnns import find_hash
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
        split_trials_driscoll,
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
        (Path(__file__).parent / '../plots/notebook/multi-pred-failure').resolve(),
        fmt = 'pdf', display=False)
    plot_root = Path(plotter.plot_dir)

    period_colors = {
        "iti": t20("b:5"),
        "context": t20("c:1"),
        "stim": t20("b:2"),
        "memory": t20("b:14"),
        "response": t20("b:10")
    }
    return colors, getc, period_colors, plot_root, plotter, styles, t20


@app.cell
def __(th):
    # cuda setup
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    cpu = th.device('cpu' if th.cuda.is_available() else 'cpu')
    print(device.type)
    return cpu, device


@app.cell
def __(mo):
    mo.md("""### Load predictions""")
    return


@app.cell
def __(
    DriscollTasks,
    MultiBlockActivityDataset,
    Path,
    __file__,
    device,
    dill,
    find_hash,
    plotter,
    rnns,
):
    root_dir = "/Users/kaifox/projects/loop/dynrn/data"
    dsn_hash = '7c61e2'

    # load discounted sum network pointed to by hash
    dsn_path = find_hash(root_dir, dsn_hash, ".pt")
    dsn_path = Path(str(dsn_path)[:-3])  # remove .pt
    final_dsn, dsn_ckpts, traindata = rnns.load_rnn(dsn_path, device=device)

    # Load associated activity dataset
    # act_data_hash = traindata["act_hash"]
    act_data_hash = '7c6042'
    act_data = dill.load(open(find_hash(root_dir, act_data_hash, ".dil"), "rb"))

    # Select main variables from activity dataset
    test_data: MultiBlockActivityDataset = act_data["test"]
    test_blocks = DriscollTasks.split_dataset(test_data)
    cumulant_fn = lambda x: x[:, 1:]
    gamma = traindata["gamma"]

    # set plotting directory
    dsn_name = dsn_path.name
    nb_plot_dir = (Path(__file__).parent / '../plots/notebook').resolve()
    plotter.plot_dir = nb_plot_dir / 'multi-pred-failure' / dsn_name
    plotter.plot_dir.mkdir(exist_ok=True)
    return (
        act_data,
        act_data_hash,
        cumulant_fn,
        dsn_ckpts,
        dsn_hash,
        dsn_name,
        dsn_path,
        final_dsn,
        gamma,
        nb_plot_dir,
        root_dir,
        test_blocks,
        test_data,
        traindata,
    )


@app.cell
def __(cumulant_fn, device, final_dsn, gamma, la, test_data, th):
    # discounted sum predictions
    x = th.tensor(test_data["activity"], dtype=th.float32, device=device)
    preds = final_dsn.to(device)(x).cpu().detach().numpy()

    # extraction of cumulant from sum
    cumul_gt = cumulant_fn(test_data["activity"])
    cumul_pr = preds[:, :-1] - gamma * preds[:, 1:]

    # timepoint-wise loss
    loss_series = la.norm(cumul_gt - cumul_pr, axis=-1)
    return cumul_gt, cumul_pr, loss_series, preds, x


@app.cell
def __(mo):
    mo.md(r"""### Loss relative to trial start""")
    return


@app.cell
def __(loss_series, periperiod_sliced, test_data):
    # indexing: period_block_loss[i_block][period_number]: DataFrame[rel_time, number, 0, 1, 2, ...]
    # where column "0" indexes last dimension of `loss_series`

    period_block_loss = {}
    for i_block, slc in enumerate(test_data['block_slices']):
        period_block_loss[i_block] = periperiod_sliced(
            loss_series[slc, :, None],
            test_data['periods'][slc, 1:].astype('int')
        )
    return i_block, period_block_loss, slc


@app.cell(disabled=True)
def __(
    colors,
    period_block_loss,
    period_colors,
    plotter,
    plt,
    test_data,
    vu,
):
    def _plot():
        npr = max(t.n_period for t in test_data['block_tasks'])
        nbl = len(test_data['block_slices'])
        fig, ax = plt.subplots(nbl, npr, figsize=(1.7 * npr, nbl * 1.7), sharey=True)
        for i_b, b_task in enumerate(test_data['block_tasks']):
            for i_p, p_name in enumerate(b_task.periods):
                block = period_block_loss[i_b][i_p]
                c = vu.lighten(period_colors[p_name], 0.5)
                a = ax[i_b, i_p]
                for k, group in block.groupby('number'):
                    a.plot(
                        group.rel_time, group[0], color=c, lw=0.05
                    )
                means = block.groupby('rel_time').mean()[0]
                a.plot(means, color=c)
                a.axvline(0, color = colors.subtle, lw = 0.5, zorder = -2)
                a.set_title(p_name, fontsize=8)
                if i_b == 0 and i_p == 0:
                    a.set_ylabel(f"{i_b} {b_task.name}\nL2 dist")
                elif i_p == 0:
                    a.set_ylabel(f"{i_b} {b_task.name}")
            for i_p in range(b_task.n_period, npr):
                ax[i_b, i_p].set_axis_off()
        ax[-1, 0].set_xlabel("steps (rel. period start)")
        return fig
    _f = _plot(); plotter.finalize(_f, 'periperiod-losses'); _f
    return


@app.cell
def __(mo):
    mo.md(r"""### Dynamics in each task at period transition""")
    return


@app.cell
def __(PCA, cumul_gt, cumul_pr, np, split_trials, test_data):
    # find task-wise, period-wise PCs

    # shorthand for pca.transform on high-dim array
    _pct = lambda pca, x: pca.transform(x.reshape(-1, x.shape[-1])).reshape(
        x.shape[:-1] + (pca.n_components_,)
    )

    def _calc(task, slc):
        # list of trials, where each trial is dict of lists of arrays
        # ex: trials[0]['period'] = [[4, 0, 1], [0, 1, 2], ..., [3, 4, 0]]
        # and trials[0]['gt'] has same list strucutre, but with period indices
        # replaced by `cumulpc_gt` data from the corresponding period
        trials_fulldim = split_trials(
            {"gt": cumul_gt[slc]},
            test_data['periods'][slc].astype("int")[:, 1:],
        )
        trials_fulldim = list(
            filter(  # filter out end-of-session ITI
                (lambda t: len(t["period"]) >= 5), trials_fulldim
            )
        )
        # fit pca for activity during each period for visualization
        _npr = task.n_period
        period_cumul_gt = [
            np.concatenate([t["gt"][i] for t in trials_fulldim]) for i in range(_npr)
        ]
        period_cumul_pca = [PCA(n_components=3).fit(h) for h in period_cumul_gt]
        full_pca = PCA(n_components=3).fit(cumul_gt.reshape(-1, cumul_gt.shape[-1]))

        # gt and predicted cumulants and predicted future sum, projected onto each PC axis
        # shape: (n_sessions, session_length, 2, n_period)
        cumulpc_gt = np.stack(
            [_pct(pca, cumul_gt[slc]) for pca in period_cumul_pca],
            axis=-1,
        )
        cumulpc_pr = np.stack(
            [_pct(pca, cumul_pr[slc]) for pca in period_cumul_pca],
            axis=-1,
        )

        cumulfpc_gt = np.stack(
            [_pct(full_pca, cumul_gt[slc]) for pca in period_cumul_pca],
            axis=-1,
        )
        cumulfpc_pr = np.stack(
            [_pct(full_pca, cumul_pr[slc]) for pca in period_cumul_pca],
            axis=-1,
        )

        return period_cumul_pca, cumulpc_gt, cumulpc_pr

    period_pca, cumulpc_gt, cumulpc_pr = {}, {}, {}
    for i, (_t, _s) in enumerate(zip(test_data['block_tasks'], test_data['block_slices'])):
        _p, _g, _r = _calc(_t, _s)
        period_pca[i] = _p
        cumulpc_gt[i] = _g
        cumulpc_pr[i] = _r
    return cumulpc_gt, cumulpc_pr, i, period_pca


@app.cell
def __(
    Colormap,
    apply_to_trial_groups,
    cumulpc_gt,
    cumulpc_pr,
    extract_trial_data,
    nanmean_with_req,
    np,
    split_trials_driscoll,
    test_blocks,
    test_data,
):
    def _calc(block, task, slc):
        # list of trials, where each trial is dict of lists of arrays
        # ex: trials['period'][0] = [[4, 0, 1], [0, 1, 2], ..., [3, 4, 0]]
        # and trials['pc_gt'][0] has same list strucutre, but with period indices
        # replaced by `cumulpc_gt` data from the corresponding period
        trials = split_trials_driscoll(
            {
                "pc_gt": cumulpc_gt[block],
                "pc_pr": cumulpc_pr[block],
                "stim": test_data['stimuli'][slc],
            },
            test_blocks[block],
            window=1,
        )

        # find stim angle for each trial and group by it
        trial_angles, trial_angle_colors, trial_groups = extract_trial_data(
            lambda x: np.arctan2(x[:, 3], x[:, 2]).mean(),
            trials,
            window = 1,
            cmap = Colormap("matlab:cool"),
            color_range = (0, np.pi / 2),
            n_clusters = 20
        )
        trial_group_angles = np.array([trial_angles[group].mean() for group in trial_groups])
        trial_group_angle_colors = Colormap("matlab:cool")(trial_group_angles / np.pi * 4)

        # average trajectories within angle groups
        trial_group_avgs = apply_to_trial_groups(
            lambda x: nanmean_with_req(x, n_req = 4, axis = 0),
            trials,
            trial_groups,
            as_array=True
        )

        return trials, trial_angle_colors, trial_group_avgs, trial_group_angle_colors


    trials, trial_colors, avg_trials, avg_trial_colors = {}, {}, {}, {}
    for _i, (_t, _s) in enumerate(zip(test_data['block_tasks'], test_data['block_slices'])):
        _t, _tc, _at, _atc = _calc(_i, _t, _s)
        trials[_i] = _t
        trial_colors[_i] = _tc
        avg_trials[_i] = _at
        avg_trial_colors[_i] = _atc
    return avg_trial_colors, avg_trials, trial_colors, trials


@app.cell
def __(Colormap, np, period_colors, test_data, vu):
    period_timepals = {
        i_b: [
            Colormap([vu.lighten(c, 0.7), c, vu.darken(c, 0.6)])(
                np.linspace(0, 1, 15)
            )
            for c in [period_colors[n] for n in t.periods]
        ]
        for i_b, t in enumerate(test_data["block_tasks"])
    }
    angle_timepal = lambda c: (
        Colormap([vu.lighten(c, 0.6), c, vu.darken(c, 0.4)])(np.linspace(0, 1, 15))
    )
    return angle_timepal, period_timepals


@app.cell
def __(
    angle_timepal,
    period_timepals,
    plotter,
    plt,
    test_data,
    trial_colors,
    trials,
    vd,
):
    def _plot():
        nbl = len(test_data['block_slices'])
        fig, axes = plt.subplots(2 * nbl, 5, figsize=(2 * 5, 2 * 3 * nbl), sharex='col', sharey='col')
        for i_b, task in enumerate(test_data['block_tasks']):
            ax = axes[2 * i_b : 2 * i_b + 2]
            angle_timepals = [angle_timepal(c) for c in trial_colors[i_b]]

            plot_n = 50
            for i_win in [1]:
                for i_per in range(5):
                    # trial array index t[key][i_phase][i_in_window][..., i_pc, i_pca]
                    tcolor = [
                        period_timepals[i_b][int(t["period"][i_per][i_win])] for t in trials[i_b]
                    ][:plot_n]
                    tdata = [t["pc_gt"][i_per][i_win][..., :2, i_per] for t in trials[i_b]][
                        :plot_n
                    ]
                    vd.trajecories(
                        ax[0, i_per],
                        tdata,
                        tcolor,
                        color="both",
                        lw=0.5,
                    )

                    vd.trajecories(
                        ax[1, i_per],
                        tdata,
                        angle_timepals[:plot_n],
                        color="both",
                        lw=0.5,
                    )

            ax[0, 0].set_ylabel(f"{i_b} {task.name}")
        return fig


    _fig = _plot()
    plotter.finalize(_fig, "true_traj_plots")
    _fig
    return


@app.cell
def __():
    return


app._unparsable_cell(
    r"""
    def _():
            
    """,
    name="__"
)


if __name__ == "__main__":
    app.run()
