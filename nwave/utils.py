'''
==========
Date: June 21, 2022
Maintainer:
Xinyi Zhong (xinyi.zhong@yale.edu)
Xinchen Du (xinchen.du@yale.edu)
Zhiyuan Long (zhiyuan.long@yale.edu)
==========
'''

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy
from configs import *
if get_argsprocessor() == "CPU":
    import numpy as cp
elif get_argsprocessor() == "GPU":
    import cupy as cp

plt.rcParams['figure.figsize'] = (8, 6)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

from matplotlib.ticker import FuncFormatter


def get_kernels(re, ri, wi=5, we=30, sigmaE = 3):
    k_exc = cp.zeros([2*re+1, 2*re+1])
    k_inh = cp.zeros([2*ri+1, 2*ri+1])
    for i in range(2*re+1):
        for j in range(2*re+1):
            # i row, j column
            distsq = (i-re)**2+(j-re)**2
            k_exc[i,j] = cp.exp(- distsq/2/sigmaE) * (distsq <= re**2)
    k_exc = we * k_exc / cp.sum(k_exc)
    for i in range(2*ri+1):
        for j in range(2*ri+1):
            # i row, j column
            distsq = (i-ri)**2+(j-ri)**2
            k_inh[i,j] = (distsq <= ri**2)
    k_inh = wi * k_inh / cp.sum(k_inh)
    return k_exc, k_inh

def vis_error(error, fpath, train_start_step=0):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(20, 3))
    ax0.plot(error[:, 0])
    ax0.set_title("reconstruction error")
    ax0.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%d" % (train_start_step + x)))
    ax1.plot(error[:, 1])
    ax1.set_title("l1norm")
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%d" % (train_start_step + x)))
    ax2.plot(error[:, 2])
    ax2.set_title("l0norm")
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%d" % (train_start_step + x)))

    fig.suptitle(fpath[10:].replace('/', ' ').strip())
    fig.savefig(fpath + 'errors{}ts.png'.format(train_start_step))
    plt.close()

def mymkdir(fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath)

def get_fpath_from_configs(cfg):
    return "../result/test{}neurons{}re{}ri{}we{}wi{}lrW{}lrA{}l0target{}threshold{}/".format(get_argsrow(), cfg.exp.neuron_shape[0]*cfg.exp.neuron_shape[1],
        cfg.kernel.re, cfg.kernel.ri, cfg.kernel.we, cfg.kernel.wi, cfg.exp.lr_codebook, cfg.exp.lr_act, cfg.exp.l0_target,
        cfg.exp.threshold)

def plot_colortable(colors, text_on=True):
    # ref: https://matplotlib.org/stable/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
    nunit = len(colors)
    side_length = int(numpy.sqrt(nunit))
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
    qtmin = numpy.quantile(vec, qt, axis=1)[:, numpy.newaxis]
    qtmax = numpy.quantile(vec, 1 - qt, axis=1)[:, numpy.newaxis]
    return numpy.minimum(numpy.maximum((vec - qtmin) / (qtmax - qtmin), 0), 1)


def get_colors(Vt, alpha=0.5):
    _, n = Vt.shape
    colors = []
    for i in range(n):
        colors.append((*Vt[:, i], alpha))
    return colors


def plot_PCA(Phi, filename=''):
    Phi = numpy.asarray(Phi)
    U, S, Vt = numpy.linalg.svd(Phi.T, full_matrices=False)   # Phi: 97 * 400
    principal_score = U @ numpy.diag(S)[:, :3]
    principal_scoreT = rescale(principal_score.T, 0.05)
    colors = get_colors(principal_scoreT, alpha=0.8)
    fig = plot_colortable(colors, text_on=False)
    if len(filename) > 0:
        fig.savefig(filename, bbox_inches='tight')
    plt.close()

