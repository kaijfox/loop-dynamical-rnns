from . import nb
import seaborn as sns
import matplotlib as mpl
import numpy as np

def dictpal(name, **kws):
    palgen = nb.palgen(name, **kws)
    def make(keys = None):
        if isinstance(keys, int):
            return np.array(palgen(keys))
        else:
            return dict(zip(keys, palgen(len(keys))))
    return make

class mph_colors(nb.colorset):
    neutral = '.1'
    semisubtle = '.7'
    subtle = '.8'
    ch0 = dictpal('ch:start=.2,rot=.3,light=.7')
    ch1 = dictpal('ch:start=.1,rot=-.1,light=.6')
    C = ['C0', 'C2', 'C3', 'C4']



def init_rc():
    mpl.rcParams.update({
        # font/text
        'font.family': 'Arial',
        'font.size': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.titlesize': 9,
        'legend.fontsize': 8,
        # axes
        'axes.facecolor': 'ffffff00',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,
        'xtick.major.size': 1.5,
        'ytick.major.size': 1.5,
        # 'xtick.direction': 'in',
        # 'ytick.direction': 'in',
        # 'axes.autolimit_mode': 'round_numbers',
        # legend
        'legend.frameon': False,
        # lines/points
        'lines.markeredgewidth': 0,
        'lines.markersize': 4,
        'lines.linewidth': 1,
        'lines.color': 'k',
    })

def init_plt(plot_dir, **kws):
    _colors, plotter = nb.init_plt(plot_dir, "default", **kws)
    sns.set_context('paper')
    init_rc()
    return mph_colors, plotter