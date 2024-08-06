import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Setup""")
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
        nanmean_with_req
    )
    from dynrn.predictors import activity_dataset, save_dsn, load_dsn, discounted_sums
    import dynrn.basic_rnns as rnns
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
        (Path(__file__).parent / '../plots/notebook/td-pred-failure').resolve(),
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


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### Load predictions""")
    return


@app.cell
def __():
    return


@app.cell
def __(Path, device, dill, getc, jit, load_dsn, t20, th):
    # Load RNN model and predictor
    root_dir = Path("/Users/kaifox/projects/loop/dynrn/data")
    model_name = "driscoll-n1024_5c135.07181253"

    # RNN model
    model_path = root_dir / "task-models" / model_name
    model = jit.load(f"{model_path}.pt", map_location=device)
    model.load_state_dict(th.load(f"{model_path}.tar", map_location=device))

    # train and validation activity dataset for predictor
    act_dataset_name = root_dir / "pred-datasets" / f"{model_name}.dil"
    act_dataset = dill.load(open(act_dataset_name, 'rb'))
    act_pca = act_dataset.train.metadata['pca']

    # discounted sum predictor model
    dsn_path = root_dir / "pred-models" / f"td-g0.9_{model_name}"
    mlp, ckpts, _train_data = load_dsn(dsn_path, device)
    cumulant_fn = _train_data['cumulant_fn']
    losses = _train_data['losses']
    gamma = _train_data['gamma']
    gamma_normalizer = sum(gamma ** i for i in range(1000))

    # plotting for driscoll memorypro task
    xcolors = [getc("k"), getc("grey"), t20("b:17"), t20("b:19")]
    ycolors = [t20("c:0"), t20("c:2")]
    period_colors = [t20("b:5"),  t20("c:1"), t20("b:2"), t20("b:14"), t20("b:10")]
    period_names = ['ITI', 'Context', 'Stim', 'Memory', 'Response']
    return (
        act_dataset,
        act_dataset_name,
        act_pca,
        ckpts,
        cumulant_fn,
        dsn_path,
        gamma,
        gamma_normalizer,
        losses,
        mlp,
        model,
        model_name,
        model_path,
        period_colors,
        period_names,
        root_dir,
        xcolors,
        ycolors,
    )


@app.cell
def __(
    act_dataset,
    cumulant_fn,
    device,
    discounted_sums,
    gamma,
    gamma_normalizer,
    la,
    mlp,
):
    # calculate discounted sum predictions
    preds = mlp(act_dataset.val.activity.to(device)).cpu().detach().numpy()

    # extraction of cumulant from sum
    cumul_gt = cumulant_fn(act_dataset.val).cpu().numpy()
    cumul_pr = preds[:, :-1] - gamma * preds[:, 1:]

    # ground truth discounted sum
    # sums[t] = h_{t+1} + gamma h_{t+2} + ...
    sums_gt = discounted_sums(cumul_gt.transpose(0, 2, 1), gamma).transpose(0, 2, 1)
    sums_pr = preds[:, :-1] / gamma_normalizer

    # timepoint-wise loss (see predictors.td_loss), shape (n_session, session_len)
    # L = || cumulant + (g * pred_{t+1}) - pred_{t+1}||
    loss_series = la.norm(cumul_gt - cumul_pr, axis=-1)
    loss_to_sum_series = la.norm(cumul_gt - sums_pr, axis=-1)
    return (
        cumul_gt,
        cumul_pr,
        loss_series,
        loss_to_sum_series,
        preds,
        sums_gt,
        sums_pr,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### Error at period transition""")
    return


@app.cell
def __(
    DriscollPlots,
    act_dataset,
    colors,
    cumul_gt,
    cumul_pr,
    loss_series,
    np,
    plotter,
    plt,
    vu,
    xcolors,
    ycolors,
):
    def session_loss_ex(session):
        fig, ax = plt.subplots(3, 1, figsize=(7, 3), sharex=True)
        for a in ax:
            DriscollPlots.memorypro(
                a,
                None,
                None,
                act_dataset.val.periods,
                xcolors,
                ycolors,
                single_ax=True,
                session=session,
            )

        _x = np.arange(1, act_dataset.val.session_length)
        ax[0].plot(_x, loss_series[session], colors.C[0], label = 'L2 dist')
        for i in range(len(ax)-1):
            ax[i+1].plot(_x, cumul_gt[session, :, i], colors.neutral, label='gt')
            ax[i+1].plot(_x, cumul_pr[session, :, i], colors.C[2], label='pred.', lw=0.7)
            ax[i+1].set_ylabel(f"dim {i}")
        vu.legend(ax[0])
        vu.legend(ax[1])
        return fig

    _fig = session_loss_ex(0); plotter.finalize(_fig, None); _fig
    return session_loss_ex,


@app.cell
def __(act_dataset, loss_series, period_start_mask, periwindows):
    period_peri_loss = [
        periwindows(
            loss_series[..., None],
            period_start_mask(act_dataset.val.periods, i)[:, 1:],
            20,
        )[..., 0]
        for i in range(int(act_dataset.val.n_period))
    ]
    return period_peri_loss,


@app.cell
def __(
    act_dataset,
    colors,
    np,
    period_colors,
    period_names,
    period_peri_loss,
    plotter,
    plt,
    vu,
):
    def period_avgs():
        npr = int(act_dataset.val.n_period)
        fig, ax = plt.subplots(1, npr, figsize=(1.7 * npr, 1.7), sharey=True)
        for i in range(npr):
            rad = (period_peri_loss[i].shape[-1] - 1) // 2
            x = np.arange(-rad, rad + 1)
            ax[i].plot(
                x, period_peri_loss[i].T, color=vu.lighten(period_colors[i], 0.5), lw=0.05
            )
            ax[i].plot(x, period_peri_loss[i].mean(axis=0), color=period_colors[i])
            ax[i].axvline(0, color = colors.subtle, lw = 0.5, zorder = -2)
            ax[i].set_title(period_names[i])
        return fig
    _f = period_avgs(); plotter.finalize(_f, 'periperiod-losses'); _f
    return period_avgs,


@app.cell
def __(act_dataset, loss_series, loss_to_sum_series, periperiod_sliced):
    period_block_loss = periperiod_sliced(
        loss_series[..., None],
        act_dataset.val.periods[:, 1:].astype('int')
    )
    period_block_loss_to_sum = periperiod_sliced(
        loss_to_sum_series[..., None],
        act_dataset.val.periods[:, 1:].astype('int')
    )
    return period_block_loss, period_block_loss_to_sum


@app.cell
def __(
    act_dataset,
    colors,
    period_block_loss,
    period_colors,
    period_names,
    plotter,
    plt,
    vu,
):
    def period_df_avgs():
        npr = int(act_dataset.val.n_period)
        fig, ax = plt.subplots(1, npr, figsize=(1.7 * npr, 1.7), sharey=True)
        for i in range(npr):
            block = period_block_loss[i]
            for k, group in block.groupby('number'):
                ax[i].plot(
                    group.rel_time, group[0], color=vu.lighten(period_colors[i], 0.5), lw=0.05
                )
            ax[i].plot(block.groupby('rel_time').mean()[0], color=period_colors[i])
            ax[i].axvline(0, color = colors.subtle, lw = 0.5, zorder = -2)
            ax[i].set_title(period_names[i])
        ax[0].set_ylabel("L2 dist")
        return fig
    _f = period_df_avgs(); plotter.finalize(_f, 'periperiod-losses-bounded'); _f
    return period_df_avgs,


@app.cell
def __(
    act_dataset,
    colors,
    period_block_loss_to_sum,
    period_colors,
    period_names,
    plotter,
    plt,
    vu,
):
    def period_df_avgs_to_sum():
        npr = int(act_dataset.val.n_period)
        fig, ax = plt.subplots(1, npr, figsize=(1.7 * npr, 1.7), sharey=True)
        for i in range(npr):
            block = period_block_loss_to_sum[i]
            for k, group in block.groupby('number'):
                ax[i].plot(
                    group.rel_time, group[0], color=vu.lighten(period_colors[i], 0.5), lw=0.05
                )
            ax[i].plot(block.groupby('rel_time').mean()[0], color=period_colors[i])
            ax[i].axvline(0, color = colors.subtle, lw = 0.5, zorder = -2)
            ax[i].set_title(period_names[i])
        ax[0].set_ylabel("L2 dist")
        return fig
    _f = period_df_avgs_to_sum(); plotter.finalize(_f, 'periperiod-losses-to-sum-bounded'); _f
    return period_df_avgs_to_sum,


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### Prediction dynamics at period transition""")
    return


@app.cell
def __(PCA, act_dataset, cumul_gt, cumul_pr, np, split_trials, sums_pr):
    # list of trials, where each trial is dict of lists of arrays
    # ex: trials[0]['period'] = [[4, 0, 1], [0, 1, 2], ..., [3, 4, 0]]
    # and trials[0]['gt'] has same list strucutre, but with period indices
    # replaced by `cumulpc_gt` data from the corresponding period
    trials_fulldim = split_trials(
        {"gt": cumul_gt},
        act_dataset.val.periods.astype("int")[:, 1:],
    )
    trials_fulldim = list(
        filter(  # filter out end-of-session ITI
            (lambda t: len(t["period"]) >= 5), trials_fulldim
        )
    )
    # fit pca for activity during each period for visualization
    _npr = int(act_dataset.val.n_period)
    period_cumul_gt = [
        np.concatenate([t["gt"][i] for t in trials_fulldim]) for i in range(_npr)
    ]
    period_cumul_pca = [PCA(n_components=3).fit(h) for h in period_cumul_gt]

    # shorthand for pca.transform on high-dim array
    _pct = lambda pca, x: pca.transform(x.reshape(-1, x.shape[-1])).reshape(
        x.shape[:-1] + (pca.n_components_,)
    )

    # gt and predicted cumulants and predicted future sum, projected onto each PC axis
    # shape: (n_sessions, session_length, 2, n_period)
    cumulpc_gt = np.stack(
        [_pct(pca, cumul_gt) for pca in period_cumul_pca],
        axis=-1,
    )
    cumulpc_pr = np.stack(
        [_pct(pca, cumul_pr) for pca in period_cumul_pca],
        axis=-1,
    )
    cumulpc_ps = np.stack(
        [_pct(pca, sums_pr) for pca in period_cumul_pca],
        axis=-1,
    )
    return (
        cumulpc_gt,
        cumulpc_pr,
        cumulpc_ps,
        period_cumul_gt,
        period_cumul_pca,
        trials_fulldim,
    )


@app.cell(hide_code=True)
def __():
    # period_cumul_gt = [
    #     cumul_gt[act_dataset.val.periods[:, :-1] == i]
    #     for i in range(int(act_dataset.val.n_period))
    # ]
    # period_cumul_pr = [
    #     cumul_pr[act_dataset.val.periods[:, :-1] == i]
    #     for i in range(int(act_dataset.val.n_period))
    # ]
    # period_cumul_pca = [PCA(n_components=2).fit(h) for h in period_cumul_gt]


    # _pct = lambda pca, x: pca.transform(x.reshape(-1, x.shape[-1])).reshape(
    #     x.shape[:-1] + (pca.n_components_,)
    # )
    # # gt and predicted cumulants, projected onto each PC axis
    # # shape: (n_sessions, session_length, 2, n_period)
    # cumulpc_gt = np.stack(
    #     [_pct(pca, cumul_gt) for pca in period_cumul_pca],
    #     axis=-1,
    # )
    # cumulpc_pr = np.stack(
    #     [_pct(pca, cumul_pr) for pca in period_cumul_pca],
    #     axis=-1,
    # )
    return


@app.cell
def __(
    Colormap,
    act_dataset,
    cumulpc_gt,
    cumulpc_pr,
    cumulpc_ps,
    extract_trial_data,
    np,
    split_trials,
):
    # list of trials, where each trial is dict of lists of arrays
    # ex: trials[0]['period'] = [[4, 0, 1], [0, 1, 2], ..., [3, 4, 0]]
    # and trials[0]['pc_gt'] has same list strucutre, but with period indices
    # replaced by `cumulpc_gt` data from the corresponding period
    trials = split_trials(
        {
            "pc_gt": cumulpc_gt,
            "pc_pr": cumulpc_pr,
            "pc_ps": cumulpc_ps,
            "stim": act_dataset.val.stim.cpu().numpy(),
        },
        act_dataset.val.periods.astype("int")[:, 1:],
        window=1,
    )
    trials = list(filter( # filter out end-of-session ITI
        (lambda t: len(t["period"]) >= 5), trials
    ))

    # find stim angle for each trial
    trial_angles, trial_angle_colors, trial_groups = extract_trial_data(
        lambda x: np.arctan2(x[:, 3], x[:, 2]).mean(),
        trials,
        window = 1,
        cmap = Colormap("matlab:cool"),
        color_range = (0, act_dataset.val.metadata["task_kws"]["angle"].args[1]),
        n_clusters = 20
    )
    trial_group_angles = np.array([trial_angles[group].mean() for group in trial_groups])
    trial_group_angle_colors = Colormap("matlab:cool")(trial_group_angles / np.pi * 4)

    return (
        trial_angle_colors,
        trial_angles,
        trial_group_angle_colors,
        trial_group_angles,
        trial_groups,
        trials,
    )


@app.cell
def __(apply_to_trial_groups, nanmean_with_req, trial_groups, trials):
    trial_group_avgs = apply_to_trial_groups(
        lambda x: nanmean_with_req(x, n_req = 4, axis = 0),
        trials,
        trial_groups,
        as_array=True
    )
    return trial_group_avgs,


@app.cell
def __(
    Colormap,
    np,
    period_colors,
    plotter,
    plt,
    trial_angle_colors,
    trials,
    vd,
    vu,
):
    def true_traj_plots():
        fig, ax = plt.subplots(2, 5, figsize=(2 * 5, 2 * 3), sharex='col', sharey='col')
        period_timepals = [
            Colormap([vu.lighten(c, 0.7), c, vu.darken(c, 0.6)])(
                np.linspace(0, 1, 15)
            )
            for c in period_colors
        ]
        angle_timepal = [
            Colormap([vu.lighten(c, 0.6), c, vu.darken(c, 0.4)])(
                np.linspace(0, 1, 15)
            )
            for c in trial_angle_colors
        ]

        plot_n = 50
        for i_win in range(3):
            for i_per in range(5):
                # trial array index t[key][i_phase][i_in_window][..., i_pc, i_pca]
                tcolor = [
                    period_timepals[t["period"][i_per][i_win]] for t in trials
                ][:plot_n]
                tdata = [t["pc_gt"][i_per][i_win][..., :2, i_per] for t in trials][
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
                    angle_timepal[:plot_n],
                    color="both",
                    lw=0.5,
                )



        plotter.finalize(fig, None)
        return fig


    _fig = true_traj_plots()
    plotter.finalize(_fig, "true_traj_plots")
    _fig
    return true_traj_plots,


@app.cell
def __(Colormap, np, period_colors, plotter, plt, trials, vd, vu):
    def true_pred_cumultraj():
        fig, ax = plt.subplots(
            4, 5, figsize=(3 * 5, 2 * 4), sharex="col", sharey="col"
        )
        period_timepals = [
            Colormap([vu.lighten(c, 0.7), c, vu.darken(c, 0.6)])(
                np.linspace(0, 1, 15)
            )
            for c in period_colors
        ]

        plot_n = 50
        for i_row, y_pc in enumerate([1, 2]):
            for i_win in range(3):
                for i_per in range(5):
                    # trial array index t[key][i_phase][i_in_window][..., i_pc]
                    tcolor = [
                        period_timepals[t["period"][i_per][i_win]] for t in trials
                    ][:plot_n]
                    
                    gtdata = [t["pc_gt"][i_per][i_win][..., [0, y_pc], i_per] for t in trials][
                        :plot_n
                    ]
                    row = 2 * i_row
                    vd.trajecories(ax[row, i_per], gtdata, tcolor, color="both", lw=0.5, time_point=10)
        
                    prdata = [t["pc_pr"][i_per][i_win][..., [0, y_pc], i_per] for t in trials][
                        :plot_n
                    ]
                    vd.trajecories(ax[row + 1, i_per], prdata, tcolor, color="both", lw=0.5)

        plotter.finalize(fig, None)
        return fig


    _fig = true_pred_cumultraj()
    plotter.finalize(_fig, "true-pred-cumultraj")
    _fig
    return true_pred_cumultraj,


@app.cell
def __(
    Colormap,
    np,
    period_colors,
    plotter,
    plt,
    trial_group_angle_colors,
    trial_group_avgs,
    vd,
    vu,
):
    def true_traj_avg_plots():
        fig, ax = plt.subplots(
            4, 5, figsize=(3 * 5, 2 * 4), sharex="col", sharey="col"
        )
        period_timepals = [
            Colormap([vu.lighten(c, 0.7), c, vu.darken(c, 0.6)])(
                np.linspace(0, 1, 15)
            )
            for c in period_colors
        ]
        angle_timepal = [
            Colormap([vu.lighten(c, 0.8), c, vu.darken(c, 0.35)])(
                np.linspace(0, 1, 15)
            )
            for c in trial_group_angle_colors
        ]

        for row, y_pc in enumerate([1, 2]):
            for i_win in range(3):
                for i_per in range(5):
        
                    # trial_avg array index t[key][i_phase][i_in_window][time, i_pc, i_pca]
                    tcolor = [
                        period_timepals[int(t["period"][i_per][i_win])]
                        for t in trial_group_avgs
                    ]
                    tdata = [
                        t["pc_gt"][i_per][i_win][..., [0, y_pc], i_per]
                        for t in trial_group_avgs
                    ]
                    vd.trajecories(ax[row, i_per], tdata, tcolor, color="both", lw=0.5)
        
                    vd.trajecories(ax[row+2, i_per], tdata, angle_timepal, color="both", lw=0.5)

                    ax[row, 0].set_ylabel(f"PC {y_pc}")
                    ax[row + 1, 0].set_ylabel(f"PC {y_pc}")
                    ax[-1, i_per].set_xlabel(f"PC 0")

        plotter.finalize(fig, None)
        return fig


    _fig = true_traj_avg_plots()
    plotter.finalize(_fig, "true-cumultraj-avg")
    _fig
    return true_traj_avg_plots,


@app.cell
def __(
    Colormap,
    np,
    period_colors,
    period_names,
    plotter,
    plt,
    trial_group_avgs,
    vd,
    vu,
):
    def traj_pred_avg_plots():
        fig, ax = plt.subplots(
            4, 5, figsize=(3 * 5, 2 * 4), sharex="col", sharey="col"
        )
        period_timepals = [
            Colormap([vu.lighten(c, 0.7), c, vu.darken(c, 0.6)])(
                np.linspace(0, 1, 15)
            )
            for c in period_colors
        ]

        for i_row, y_pc in enumerate([1, 2]):
            row = 2 * i_row
            for i_win in range(3):
                for i_per in range(5):
                    # trial_avg array index t[key][i_phase][i_in_window][time, i_pc, i_pca]
                    tcolor = [
                        period_timepals[int(t["period"][i_per][i_win])]
                        for t in trial_group_avgs
                    ]
                    gtdata = [
                        t["pc_gt"][i_per][i_win][..., [0, y_pc], i_per]
                        for t in trial_group_avgs
                    ]
                    prdata = [
                        t["pc_pr"][i_per][i_win][..., [0, y_pc], i_per]
                        for t in trial_group_avgs
                    ]
                    vd.trajecories(
                        ax[row, i_per], gtdata, tcolor, color="both", lw=0.75
                    )

                    vd.trajecories(
                        ax[row+1, i_per], prdata, tcolor, color="both", lw=0.75
                    )

                    ax[row, 0].set_ylabel(f"PC {y_pc}  |  true")
                    ax[row + 1, 0].set_ylabel(f"PC {y_pc}  |  pred.")
                    ax[-1, i_per].set_xlabel(f"PC 0")
                    ax[0, i_per].set_title(period_names[i_per])

        plotter.finalize(fig, None)
        return fig


    _fig = traj_pred_avg_plots()
    plotter.finalize(_fig, "true-pred-cumultraj-avg")
    _fig
    return traj_pred_avg_plots,


@app.cell
def __(
    Colormap,
    np,
    period_names,
    plotter,
    plt,
    trial_group_angle_colors,
    trial_group_avgs,
    vd,
    vu,
):
    def traj_pred_avg_plots_angle():
        fig, ax = plt.subplots(
            4, 5, figsize=(3 * 5, 2 * 4), sharex="col", sharey="col"
        )
        angle_timepal = [
            Colormap([vu.lighten(c, 0.8), c, vu.darken(c, 0.35)])(
                np.linspace(0, 1, 15)
            )
            for c in trial_group_angle_colors
        ]

        for i_row, y_pc in enumerate([1, 2]):
            row = 2 * i_row
            for i_win in range(3):
                for i_per in range(5):
                    # trial_avg array index t[key][i_phase][i_in_window][time, i_pc, i_pca]
                    gtdata = [
                        t["pc_gt"][i_per][i_win][..., [0, y_pc], i_per]
                        for t in trial_group_avgs
                    ]
                    prdata = [
                        t["pc_pr"][i_per][i_win][..., [0, y_pc], i_per]
                        for t in trial_group_avgs
                    ]
                    vd.trajecories(
                        ax[row, i_per], gtdata, angle_timepal, color="both", lw=0.75
                    )

                    vd.trajecories(
                        ax[row+1, i_per], prdata, angle_timepal, color="both", lw=0.75
                    )

                    ax[row, 0].set_ylabel(f"PC {y_pc}  |  true")
                    ax[row + 1, 0].set_ylabel(f"PC {y_pc}  |  pred.")
                    ax[-1, i_per].set_xlabel(f"PC 0")
                    ax[0, i_per].set_title(period_names[i_per])

        plotter.finalize(fig, None)
        return fig


    _fig = traj_pred_avg_plots_angle()
    plotter.finalize(_fig, "true-pred-cumultraj-avg-angle")
    _fig
    return traj_pred_avg_plots_angle,


@app.cell
def __(
    Colormap,
    np,
    period_colors,
    period_names,
    plotter,
    plt,
    trial_group_avgs,
    vd,
    vu,
):
    def traj_predsum_avg_plots():
        fig, ax = plt.subplots(
            4, 5, figsize=(3 * 5, 2 * 4), sharex="col", sharey="col"
        )
        period_timepals = [
            Colormap([vu.lighten(c, 0.7), c, vu.darken(c, 0.6)])(
                np.linspace(0, 1, 15)
            )
            for c in period_colors
        ]

        for i_row, y_pc in enumerate([1, 2]):
            row = 2 * i_row
            for i_win in range(3):
                for i_per in range(5):
                    # trial_avg array index t[key][i_phase][i_in_window][time, i_pc, i_pca]
                    tcolor = [
                        period_timepals[int(t["period"][i_per][i_win])]
                        for t in trial_group_avgs
                    ]
                    gtdata = [
                        t["pc_gt"][i_per][i_win][..., [0, y_pc], i_per]
                        for t in trial_group_avgs
                    ]
                    psdata = [
                        t["pc_ps"][i_per][i_win][..., [0, y_pc], i_per]
                        for t in trial_group_avgs
                    ]
                    vd.trajecories(
                        ax[row, i_per], gtdata, tcolor, color="both", lw=0.75
                    )

                    vd.trajecories(
                        ax[row+1, i_per], psdata, tcolor, color="both", lw=0.75
                    )

                    ax[row, 0].set_ylabel(f"PC {y_pc}  |  true")
                    ax[row + 1, 0].set_ylabel(f"PC {y_pc}  |  pred. sum")
                    ax[-1, i_per].set_xlabel(f"PC 0")
                    ax[0, i_per].set_title(period_names[i_per])

        plotter.finalize(fig, None)
        return fig


    _fig = traj_predsum_avg_plots()
    plotter.finalize(_fig, "true-pred-sumtraj-avg")
    _fig
    return traj_predsum_avg_plots,


@app.cell
def __(
    Colormap,
    np,
    period_names,
    plotter,
    plt,
    trial_group_angle_colors,
    trial_group_avgs,
    vd,
    vu,
):
    def traj_pred_avg_plots_angle_sum():
        fig, ax = plt.subplots(
            4, 5, figsize=(3 * 5, 2 * 4), sharex="col", sharey="col"
        )
        angle_timepal = [
            Colormap([vu.lighten(c, 0.8), c, vu.darken(c, 0.35)])(
                np.linspace(0, 1, 15)
            )
            for c in trial_group_angle_colors[::2]
        ]

        for i_row, y_pc in enumerate([1, 2]):
            row = 2 * i_row
            for i_win in range(3):
                for i_per in range(5):
                    # trial_avg array index t[key][i_phase][i_in_window][time, i_pc, i_pca]
                    gtdata = [
                        t["pc_gt"][i_per][i_win][..., [0, y_pc], i_per]
                        for t in trial_group_avgs[::2]
                    ]
                    prdata = [
                        t["pc_ps"][i_per][i_win][..., [0, y_pc], i_per]
                        for t in trial_group_avgs[::2]
                    ]
                    vd.trajecories(
                        ax[row, i_per], gtdata, angle_timepal, color="both", lw=0.75
                    )

                    vd.trajecories(
                        ax[row+1, i_per], prdata, angle_timepal, color="both", lw=0.75
                    )

                    ax[row, 0].set_ylabel(f"PC {y_pc}  |  true")
                    ax[row + 1, 0].set_ylabel(f"PC {y_pc}  |  pred.")
                    ax[-1, i_per].set_xlabel(f"PC 0")
                    ax[0, i_per].set_title(period_names[i_per])

        plotter.finalize(fig, None)
        return fig


    _fig = traj_pred_avg_plots_angle_sum()
    plotter.finalize(_fig, "true-pred-sumtraj-avg-angle")
    _fig
    return traj_pred_avg_plots_angle_sum,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
