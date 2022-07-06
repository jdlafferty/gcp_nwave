#################################
# Read data from CSV
#################################

from csv import DictReader

import math
import argparse

def parse_argv():
    parser = argparse.ArgumentParser(prog='SC')
    parser.add_argument("--row", type = int)
    parser.add_argument("--processor", type = str)
    return parser.parse_args()

args = parse_argv()

params = []
with open('parameter.csv', 'r') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
    csv_dict_reader = DictReader(read_obj)
    # iterate over each line as a ordered dictionary
    for row in csv_dict_reader:
        # row variable is a dictionary that represents a row in csv
        params.append(row)


i = args.row
param = params[i-1]

def get_argsrow():
    return i

def get_argsprocessor():
    return args.processor

def get_neuron_shape(x):
    y = int(math.sqrt(x))
    return (y, y)

ri = int(param['ri'])
re = int(param['re'])
wi = int(param['wi'])
we = int(param['we'])
leaky = wi + we
input_dim = 97
neuron_shape = get_neuron_shape(int(param['neuron_shape']))
gradient_steps = int(param['gradient_steps'])
batch_size = int(param['batch_size'])
lr_act = float(param['lr_act'])
lr_codebook = float(param['lr_codebook'])
l0_target = float(param['l0_target'])
threshold = float(param['threshold'])

#################################
# Utils
#################################

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy

if get_argsprocessor() == "CPU":
    import numpy as cp
    from scipy.signal import convolve
elif get_argsprocessor() == "GPU":
    import cupy as cp
    from cusignal.convolution.convolve import convolve

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

def mymkdir(fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath)

def get_fpath_from_configs():
    return "../result/test{}neurons{}re{}ri{}we{}wi{}lrW{}lrA{}l0target{}threshold{}/".format(get_argsrow(), neuron_shape[0]*neuron_shape[1],
        re, ri, we, wi, lr_codebook, lr_act, l0_target, threshold)

def fpath_compute_act():
    return "../result/compute_act{}/".format(get_argsrow())

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
    if get_argsprocessor()  == "GPU":
        Phi = cp.asnumpy(Phi)
    U, S, Vt = numpy.linalg.svd(Phi.T, full_matrices=False)   # Phi: 97 * 400
    principal_score = U @ numpy.diag(S)[:, :3]
    principal_scoreT = rescale(principal_score.T, 0.05)
    colors = get_colors(principal_scoreT, alpha=0.8)
    fig = plot_colortable(colors, text_on=False)
    if len(filename) > 0:
        fig.savefig(filename, bbox_inches='tight')
    plt.close()

#################################
# Dataloader
#################################
cnt = 0
word_embeddings = numpy.load('../data/googleNgram/embed100.npy')
word_embeddings = numpy.delete(word_embeddings, [55, 58, 84], axis = 1)
word_embeddings = cp.asarray(word_embeddings)
word_freq = numpy.load("../data/googleNgram/1gramSortedFreq.npy")
num_train_vocabs = word_freq.shape[0]
num_test_vocabs = numpy.asarray(20000)
SUBSAMPLE_SIZE = numpy.asarray(4096)

def load_test_batch():
    global cnt
    if cnt > num_test_vocabs - batch_size:
        cnt = num_test_vocabs - batch_size
    idx = cp.arange(cnt, cnt + batch_size)
    cnt += batch_size
    word_batch = word_embeddings[idx, :]
    return word_batch, idx

#####################################
# Algorithms to update activations
#####################################
exck, inhk = get_kernels(re = re, ri = ri, we = we, wi = wi)
bs = batch_size
exc_act = cp.zeros(shape = (bs, neuron_shape[0], neuron_shape[1]))   # shape should be (bs, neuron_shape)!
inh_act = cp.zeros(shape = (bs, neuron_shape[0], neuron_shape[1]))
exck = cp.expand_dims(cp.asarray(exck), axis = 0)
inhk = cp.expand_dims(cp.asarray(inhk), axis = 0)
lr_act = cp.asarray(lr_act)
l0_target = cp.asarray(l0_target)
leaky = cp.asarray(leaky)
max_act_fit = cp.asarray(50)
threshold = threshold
eps = cp.asarray(5e-3)

def perceive_to_get_stimulus(word_batch, codebook):
    stimulus = cp.dot(word_batch, codebook).reshape((word_batch.shape[0], neuron_shape[0], neuron_shape[1]))  # word_batch = this_X = (256, 97), code_book = (97, 400)
    return stimulus   # shape: (256, 400)

def stimulate(stimulus):  # stimulus: (256, 20, 20)
    global exc_act
    global inh_act
    for t in range(int(max_act_fit)):
        exc_act_tm1 = cp.copy(exc_act)
        exc_input = convolve(exc_act, exck, mode="same")  # (256, 20, 20)
        inh_input = convolve(inh_act, inhk, mode="same")
        exc_act = exc_act + lr_act * (- leaky * exc_act + stimulus + exc_input - inh_input)
        exc_act = cp.maximum(exc_act - threshold, 0) - cp.maximum(-exc_act - threshold, 0)

        da = exc_act - exc_act_tm1
        relative_error = cp.sqrt(cp.square(da).sum()) / (eps + cp.sqrt(cp.square(exc_act_tm1).sum()))

    if relative_error < eps:
        return exc_act
    else:
        print("Relative error end with {:.4f} and doesn't converge within the max fit steps".format(exc_act))
        return exc_act


#################################
# Compute and Plot activations
#################################
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable

fpath = fpath_compute_act()
mymkdir(fpath)
Fpath = get_fpath_from_configs()
Phi = cp.load(get_fpath_from_configs() + "codebook.npy")

emb_dim, num_units = Phi.shape
activity = cp.zeros(shape=(num_test_vocabs, num_units))

for t in range((num_test_vocabs - 1) // batch_size + 1):  # batch_size = 256
    word_batch, wp_idx = load_test_batch()
    try:
        stimulus = perceive_to_get_stimulus(word_batch, Phi)
        activ = stimulate(stimulus)
        activ = activ.reshape([256, num_units])
        activity[wp_idx, :] = activ
    except RuntimeError as e:
        print(e)

with open(fpath + '/uactivity.npy', 'wb') as f:
    cp.save(f, activity)

activity = numpy.load(fpath + "uactivity.npy")
with open('../data/googleNgram/4vocabidx.pkl', 'rb') as f:
    vocabidx = pickle.load(f)

def plot_word_activations(words, filename=''):
    i = 0
    for word in words:
        try:
            activ = activity[vocabidx[word]]
        except Exception:
            print("word: {} not found".format(word))
        else:
            fig, ax = plt.subplots(figsize=(5, 5))
            l0norm = numpy.abs(activ).max()
            im = ax.imshow(activ.reshape(neuron_shape[0], neuron_shape[1]),
                               cmap='jet', interpolation='gaussian', vmin=-l0norm, vmax=l0norm)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.2)
            plt.colorbar(im, cax=cax)
            ax.set_title("{}".format(word), fontsize=24)
            ax.set_axis_off()
            if len(filename) > 0:
                plt.savefig(fpath + '%s_%d.pdf' % (filename, i))
                i = i + 1



if __name__ == "__main__":
    plot_word_activations(['technology', 'microsoft', 'intel', 'ibm', 'apple', 'banana'], 'tech')

