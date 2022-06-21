'''
==========
Date: June 15, 2022
Maintainer:
Xinyi Zhong (xinyi.zhong@yale.edu)
Xinchen Du (xinchen.du@yale.edu)
Zhiyuan Long (zhiyuan.long@yale.edu)
==========
'''

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams['figure.figsize'] = (8, 6)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

#################################
# Experiment Setup
#################################

#import cupy as cp
import numpy as cp
import numpy as np
from configs import cfg

from nwave.dictlearner import REGISTRY as dlRegistry
DL = dlRegistry[cfg.exp.dl_name]

from nwave.neurondynamics import REGISTRY as ndmRegistry
NDM = ndmRegistry[cfg.exp.ndm_name]

from nwave.utils import get_kernels
k_exc, k_inh = get_kernels(re = cfg.kernel.re, ri = cfg.kernel.ri, \
   we = cfg.kernel.we, wi = cfg.kernel.wi)

fpath = "result/"


#################################
# Plotting func
#################################


def plot_colortable(colors, text_on=True):
    # ref: https://matplotlib.org/stable/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
    nunit = len(colors)
    side_length = int(np.sqrt(nunit))
    swatch_width = cell_width = cell_height = 32
    # set figs
    ncols = nrows = side_length
    width = cell_width * ncols
    height = cell_height * nrows
    dpi = 72
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    # set ax axis
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows), 0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for unit_idx, nunit in enumerate(range(nunit)):
        row = unit_idx // ncols
        col = unit_idx % ncols
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col  # + swatch_width

        if text_on:
            ax.text(text_pos_x + cell_width / 2, y + cell_height / 2, unit_idx, fontsize=6,
                    horizontalalignment='center', verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y), width=swatch_width, height=swatch_width, facecolor=colors[unit_idx],
                      edgecolor='0.7')
            )

    return fig


def rescale(vec, qt):
    qtmin = np.quantile(vec, qt, axis=1)[:, np.newaxis]
    qtmax = np.quantile(vec, 1 - qt, axis=1)[:, np.newaxis]
    return np.minimum(np.maximum((vec - qtmin) / (qtmax - qtmin), 0), 1)


def get_colors(Vt, alpha=0.5):
    _, n = Vt.shape
    colors = []
    for i in range(n):
        colors.append((*Vt[:, i], alpha))
    return colors


def plot_PCA(Phi, filename=''):
    #print(Phi)
    U, S, Vt = np.linalg.svd(Phi.T, full_matrices=False)   # Phi: 97 * 400
    #print(S)
    principal_score = U @ np.diag(S)[:, :3]
    principal_scoreT = rescale(principal_score.T, 0.05)
    colors = get_colors(principal_scoreT, alpha=0.8)
    fig = plot_colortable(colors, text_on=False)
    if len(filename) > 0:
        fig.savefig(filename, bbox_inches='tight')
    #fig.show()
    plt.close()


#################################
# Compute Receptive fields
#################################




if __name__ == "__main__":

    Phi = cp.load(fpath + "codebook.npy")

    emb_dim, num_units = Phi.shape  # 97 * 400

    dl = DL(input_dim = cfg.exp.input_dim, neuron_shape = cfg.exp.neuron_shape, lr_codebook = cfg.exp.lr_codebook)
    ndm = NDM(neuron_shape = cfg.exp.neuron_shape, leaky = cfg.kernel.leaky, exck = k_exc, inhk = k_inh,
         lr_act = cfg.exp.lr_act, l0_target = cfg.exp.l0_target, threshold = cfg.exp.threshold, bs = 97)

    batch = np.eye(emb_dim)
    batch = batch - np.mean(batch, axis=1)
    batch = batch / np.std(batch, axis=1)

    dl.codebook = Phi
    stimulus = dl.perceive_to_get_stimulus(batch)
    RC = ndm.stimulate(stimulus)
    RC = RC.reshape([emb_dim, num_units])

    np.save(fpath + "receptive_fields.npy", RC)

    plot_PCA(RC, fpath + 'RC.pdf')


