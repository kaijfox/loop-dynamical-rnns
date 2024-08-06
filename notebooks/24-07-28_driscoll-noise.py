import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Setup and import""")
    return


@app.cell
def __():
    from dynrn.rnntasks import DriscollTasks, itiexp, DriscollPlots
    from dynrn.task_noise import (
        normalize_range, normalize_stats, lowpass_noise, get_limits, get_stats
    )
    from dynrn.predictors import (activity_dataset, load_dsn, td_loss)
    from mplutil import util as vu

    import torch as th
    import torch.jit as jit
    import copy
    import numpy as np
    import dill
    from collections import namedtuple
    from torch import nn
    import scipy.stats
    from scipy.stats import uniform, norm
    from datetime import datetime

    import matplotlib.pyplot as plt
    from pathlib import Path
    import tqdm
    import os
    import joblib as jl
    import time
    import seaborn as sns
    import marimo as mo
    return (
        DriscollPlots,
        DriscollTasks,
        Path,
        activity_dataset,
        copy,
        datetime,
        dill,
        get_limits,
        get_stats,
        itiexp,
        jit,
        jl,
        load_dsn,
        lowpass_noise,
        mo,
        namedtuple,
        nn,
        norm,
        normalize_range,
        normalize_stats,
        np,
        os,
        plt,
        scipy,
        sns,
        td_loss,
        th,
        time,
        tqdm,
        uniform,
        vu,
    )


@app.cell
def __(Path, __file__):
    from dynrn.viz import styles
    from dynrn.viz.styles import getc
    t20 = lambda x: getc(f"seaborn:tab20{x}")
    colors, plotter = styles.init_plt(
        (Path(__file__).parent / '../plots/notebook/driscoll-noise').resolve(),
        fmt = 'pdf',
        display=False)
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
    mo.md("""### Calculate and display noise types""")
    return


@app.cell
def __(DriscollTasks, Path, get_limits, get_stats, getc, jl, t20):
    root_dir = Path("/Users/kaifox/projects/loop/dynrn/data")
    task_kws = jl.load(root_dir / 'tasks' / "memorypro-long.jl")

    session_length = 500
    n_sessions = 50
    ex_seed = 1520
    sample_x, sample_y, periods = DriscollTasks.memorypro(
        **{
            **task_kws,
            **dict(
                session_length=500,
                n_sessions=n_sessions,
                seed=ex_seed,
            )
        }
    )

    xcolors = [getc("k"), getc("grey"), t20("b:17"), t20("b:19")]
    ycolors = [t20("c:0"), t20("c:2")]

    target_range = get_limits(sample_x, axis = -1)
    target_stats = get_stats(sample_x, axis = -1)
    return (
        ex_seed,
        n_sessions,
        periods,
        root_dir,
        sample_x,
        sample_y,
        session_length,
        target_range,
        target_stats,
        task_kws,
        xcolors,
        ycolors,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md("""##### Low frequency noise""")
    return


@app.cell
def __(
    lowpass_noise,
    n_sessions,
    normalize_range,
    normalize_stats,
    sample_x,
    session_length,
    target_range,
    target_stats,
):
    lowpass_freqs = [2, 5, 10, 20, 40]
    lowpass_signals = [
        lowpass_noise(
            f,
            session_length,
            session_length,
            size=(n_sessions, sample_x.shape[-1]),
        ).transpose(0, 2, 1)
        for f in lowpass_freqs
    ]
    lowpass_signals_range = [
        normalize_range(sig, target_range, axis = -1)
        for sig in lowpass_signals
    ]
    lowpass_signals_stats = [
        normalize_stats(sig, target_stats, axis = -1)
        for sig in lowpass_signals
    ]
    return (
        lowpass_freqs,
        lowpass_signals,
        lowpass_signals_range,
        lowpass_signals_stats,
    )


@app.cell
def __(
    DriscollPlots,
    lowpass_freqs,
    lowpass_signals_range,
    plotter,
    plt,
    xcolors,
    ycolors,
):
    _fig, _ax = plt.subplots(2, 2, figsize = (7, 2))
    DriscollPlots.memorypro(_ax[:, 0], lowpass_signals_range[0], None, None, xcolors, ycolors)
    DriscollPlots.memorypro(_ax[:, 1], lowpass_signals_range[2], None, None, xcolors, ycolors, legend=True)
    _fig.suptitle("Lowpass noise - range normalized")
    _ax[0, 0].set_title(f"Frequency: {lowpass_freqs[0]}")
    _ax[0, 1].set_title(f"Frequency: {lowpass_freqs[2]}")
    plotter.finalize(_fig, 'noise-lp-range', display=False)
    _fig
    return


@app.cell
def __(
    DriscollPlots,
    lowpass_freqs,
    lowpass_signals_stats,
    plotter,
    plt,
    xcolors,
    ycolors,
):
    _fig, _ax = plt.subplots(2, 2, figsize = (7, 2))
    DriscollPlots.memorypro(_ax[:, 0], lowpass_signals_stats[0], None, None, xcolors, ycolors)
    DriscollPlots.memorypro(_ax[:, 1], lowpass_signals_stats[2], None, None, xcolors, ycolors, legend=True)
    _fig.suptitle("Lowpass noise - stat. normalized")
    _ax[0, 0].set_title(f"Frequency: {lowpass_freqs[0]}")
    _ax[0, 1].set_title(f"Frequency: {lowpass_freqs[2]}")
    plotter.finalize(_fig, 'noise-lp-stat', display=False)
    _fig
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""##### Extended periods""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""##### Assemble all noise types to dictionary""")
    return


@app.cell
def __(lowpass_freqs, lowpass_signals_range, lowpass_signals_stats, np):
    noise = {
        **{f"lo-{f}": s for f, s in zip(lowpass_freqs, lowpass_signals_stats)},
        ** {
            f"lo-{f}_flag": s * np.array([1, 1, 0, 0])[None, None]
            for f, s in zip(lowpass_freqs, lowpass_signals_stats)
        },
        ** {
            f"lo-{f}_stim": s * np.array([0, 0, 1, 1])[None, None]
            for f, s in zip(lowpass_freqs, lowpass_signals_stats)
        }
    }
    noise_rangenorm = {
        **{
            f"lo-{f}_range": s
            for f, s in zip(lowpass_freqs, lowpass_signals_range)
        }
    }
    noise = {**noise, **noise_rangenorm}
    return noise, noise_rangenorm


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### Predict dynamics under noise""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""##### Load RNN and hidden trajectories""")
    return


@app.cell
def __(device, dill, jit, root_dir, th):
    model_name = "driscoll-n1024_5c135.07181253"

    model_path = root_dir / "task-models" / model_name
    model = jit.load(f"{model_path}.pt", map_location=device)
    model.load_state_dict(th.load(f"{model_path}.tar", map_location=device))

    pred_dataset_name = root_dir / "pred-datasets" / "driscoll-n1024_5c135.07181253.dil"
    pred_dataset = dill.load(open(pred_dataset_name, 'rb'))
    pred_pca = pred_dataset.train.metadata['pca']

    return (
        model,
        model_name,
        model_path,
        pred_dataset,
        pred_dataset_name,
        pred_pca,
    )


@app.cell
def __(root_dir):
    activity_path = root_dir / "pred-datasets" / "noise-5c135.07181253.dil"
    return activity_path,


@app.cell
def __(Path, activity_path, dill):
    # Load model if it has already been saved
    if not Path(f"{activity_path}").exists():
        print(f"No file: {activity_path}")
    else:
        _activity_pkl = dill.load(open(activity_path, 'rb'))
        driven_activity = _activity_pkl['driven_activity']
        print(f"Loaded activity dataset: {activity_path}")
    return driven_activity,


@app.cell(hide_code=True)
def __(mo):
    mo.md("""##### Local cell: calculate driven RNN activity for each stimulus set""")
    return


@app.cell(disabled=True)
def __(
    activity_dataset,
    activity_path,
    device,
    dill,
    model,
    noise,
    pred_pca,
    th,
    tqdm,
):
    _driven_activity = {
        k: activity_dataset(
            stim=x,
            targets=None,
            periods=None,
            activity=pred_pca.transform(
                model.seq_forward(
                    th.tensor(x, dtype=th.float32, device=device),
                    th.zeros(
                        x.shape[0],
                        model.nh,
                    ),
                )[1]
                .cpu()
                .detach()
                .numpy()
                .reshape(-1, model.nh)
            ).reshape(x.shape[:-1] + (pred_pca.n_components,)),
            n_stim=x.shape[-1],
            n_target=None,
            n_period=None,
            n_act=pred_pca.n_components,
            session_length=x.shape[1],
            n_session=x.shape[0],
            metadata={"noise_type": k},
        )
        for k, x in tqdm.tqdm(noise.items(), total=len(noise))
    }

    dill.dump(
        {
            "driven_activity": _driven_activity,
            "source": "notebooks/driscoll-noise.py",
        },
        open(activity_path, 'wb')
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""##### Load and run predcitor network""")
    return


@app.cell
def __(Path, device, load_dsn, model_name, root_dir):
    dsn_path = root_dir / "pred-models" / f"td-g0.9_{model_name}"
    if not Path(f"{dsn_path}.pt").exists():
        print(f"No such model: {dsn_path}")

    else:
        mlp, ckpts, _train_data = load_dsn(dsn_path, device)
        cumulant_fn = _train_data['cumulant_fn']
        losses = _train_data['losses']
        gamma = _train_data['gamma']
        print(f"Loaded DSN model: {dsn_path}")
    return ckpts, cumulant_fn, dsn_path, gamma, losses, mlp


@app.cell
def __(cumulant_fn, device, driven_activity, gamma, mlp, td_loss, th):
    noise_predictions = {
        k: mlp(th.tensor(d.activity, dtype=th.float32, device=device)).cpu().detach().numpy()
        for k, d in driven_activity.items()
    }
    noise_losses = {
        k: td_loss(th.tensor(sums_pred), th.tensor(cumulant_fn(driven_activity[k])), gamma)
        for k, sums_pred in noise_predictions.items()
    }
    return noise_losses, noise_predictions


@app.cell
def __(colors, losses, noise, noise_losses, plotter, plt):
    _fig, _ax = plt.subplots(1, 1, figsize = (len(noise) * 0.18, 2))
    _k = list(noise.keys())
    _x = list(range(len(_k)))
    _ax.bar(_x, [noise_losses[k] for k in _k], color = colors.subtle)
    _ax.bar([-1], [losses[-1]], color = colors.C[0])
    _ax.set_xticks([-1] + _x)
    _ax.set_xticklabels(['train'] + _k, rotation = 80)
    _ax.set_ylabel("Mean pred. error (L2)")
    plotter.finalize(_fig, 'mean-pred-errors'); _fig

    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
