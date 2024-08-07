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
    import tqdm
    import os
    import joblib as jl
    import time
    import glob
    import seaborn as sns
    return (
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
    t20 = lambda x: getc(f"seaborn:tab20{x}")
    colors, plotter = styles.init_plt(
        '../plots/notebook/first-layer-analysis',
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
    mo.md("""### Load dataset and predictor""")
    return


@app.cell
def __(Path, find_hash, jl, load_dsn):
    root_dir = Path("/Users/kaifox/projects/loop/dynrn/data")

    dset_hash = '7add88'
    dset_path = find_hash(root_dir, dset_hash, '.dil')
    data = jl.load(dset_path)['dataset']
    print("Loaded dataset:", dset_path)

    dsn_hash = '7add5b' # old bug that named model file .dil.pt
    dsn_path = find_hash(root_dir, dsn_hash, '.pt')
    mlp, ckpts, train_data = load_dsn(dsn_path)
    print("Loaded DSN model:", dsn_path)
    return (
        ckpts,
        data,
        dset_hash,
        dset_path,
        dsn_hash,
        dsn_path,
        mlp,
        root_dir,
        train_data,
    )


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ### Measure PCs of dataset and their alignment first DSN layer

        This mostly makes sense for a DSN in which the first layer is a large dimensionality reduction of previous layers, e.g. when no PCA was applied in generating its base dataset.
        """
    )
    return


@app.cell
def __(PCA, data):
    pcs = PCA().fit(data.train.activity.cpu().numpy().reshape(-1, data.train.n_act))
    return pcs,


@app.cell
def __(cosine_dist, mlp, np, pcs, scipy):
    # calculate alignment of each PC with non-null (aka row) space of weight matrix
    _weights = list(mlp.children())[0].weight.detach()

    def component_weights_onto_subspace(subspace_rows, components):
        # projection onto row space (n_hidden x n_hidden), rank n_reduced
        _proj = _weights.T @ scipy.linalg.pinv(_weights.T)
        _components = pcs.components_.T
        # project the components (n_hidden x n_components)
        _projected = _proj @ _components
        # cosine of each compone to projection onto row space
        return np.array([
            1 - cosine_dist(_projected[:, i_comp], pcs.components_[i_comp])
            for i_comp in range(pcs.components_.shape[1])
        ])

    component_weight_dists = component_weights_onto_subspace(_weights, pcs.components_)

    return component_weight_dists, component_weights_onto_subspace


@app.cell
def __(component_weight_dists, dsn_hash, plotter, plt):
    _fig, _ax = plt.subplots(figsize = (2, 2))
    _ax.plot(component_weight_dists[:80])
    _ax.set_ylabel("Cos. to learned subspace")
    _ax.set_xlabel("PC")
    plotter.finalize(_fig, f"cos-to-proj_{dsn_hash}")
    _fig
    return


@app.cell
def __(
    component_weights_onto_subspace,
    dill,
    find_hash,
    pcs,
    plotter,
    plt,
    root_dir,
):
    _fig, _ax = plt.subplots(figsize = (2, 2))
    for _hash in ['7b017e', '7b01e2', '7b01e2.0']:
        _res_path = find_hash(root_dir / 'intermediates/bottleneck', _hash, '.dil')
        _weights = dill.load(open(_res_path, 'rb'))['proj_weights']
        print(_res_path)
        _dists = component_weights_onto_subspace(_weights, pcs.components_)
        _ax.plot(_dists[:80])
    _ax.set_ylabel("Cos. to learned subspace")
    _ax.set_xlabel("PC")
    plotter.finalize(_fig, f"cos-to-proj_b40")
    _fig
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
