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
print("ri = " +str(ri))
re = int(param['re'])
print("re = " +str(re))
wi = int(param['wi'])
print("wi = " +str(wi))
we = int(param['we'])
print("we = " +str(we))
leaky = wi + we
input_dim = 97
neuron_shape = get_neuron_shape(int(param['neuron_shape']))
print("neuron_shape = " +str(neuron_shape))
gradient_steps = int(param['gradient_steps'])

lr_act = float(param['lr_act'])
print("lr_act = " +str(lr_act))
lr_codebook = float(param['lr_codebook'])
l0_target = float(param['l0_target'])
threshold = float(param['threshold'])
print("threshold = " +str(threshold))

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

def mymkdir(fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath)

def get_fpath_from_configs():
    return "../result/test{}neurons{}re{}ri{}we{}wi{}lrW{}lrA{}l0target{}threshold{}/".format(get_argsrow(), neuron_shape[0]*neuron_shape[1],
        re, ri, we, wi, lr_codebook, lr_act, l0_target, threshold)

def fpath_compute_act():
    return "../result/row{}_compute_act_laplacian/".format(get_argsrow())

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
num_test_vocabs = numpy.asarray(20000)
SUBSAMPLE_SIZE = numpy.asarray(4096)

def load_test_batch(words):
    idx = []
    for word in words:
        idx.append(vocabidx[word])
    word_batch = word_embeddings[idx, :]
    return word_batch, idx

#####################################
# Algorithms to update activations
#####################################

lr_act = cp.asarray(lr_act)
l0_target = cp.asarray(l0_target)
leaky = cp.asarray(leaky)
max_act_fit = cp.asarray(50)
threshold = threshold
eps = cp.asarray(5e-3)

def get_kernels_sum(re, ri, wi=5, we=30, sigmaE = 3):
    k_exc = cp.zeros([2*re+1, 2*re+1])
    k_inh = cp.zeros([2*ri+1, 2*ri+1])
    for i in range(2*re+1):
        for j in range(2*re+1):
            # i row, j column
            distsq = (i-re)**2+(j-re)**2
            k_exc[i,j] = cp.exp(- distsq/2/sigmaE) * (distsq <= re**2)
    sum_exc = cp.sum(k_exc)
    k_exc = we * k_exc / sum_exc
    for i in range(2*ri+1):
        for j in range(2*ri+1):
            # i row, j column
            distsq = (i-ri)**2+(j-ri)**2
            k_inh[i,j] = (distsq <= ri**2)
    sum_inh = cp.sum(k_inh)
    k_inh = wi * k_inh / sum_inh
    return k_exc, k_inh, sum_exc, sum_inh

k_exc, k_inh, sum_exc, sum_inh = get_kernels_sum(re, ri)

def get_laplacian(n, r1, r2, wi, we, sigmaE = 3):
    assert r1 < r2
    r1sq = r1**2
    r2sq = r2**2
    side_length = int(cp.sqrt(n))
    def find_distsq(i,j):
        xi, yi = i//side_length, i%side_length
        xj, yj = j//side_length, j%side_length
        return (xi-xj)**2+(yi-yj)**2
    # construct the W matrix
    We = cp.zeros(shape = (n,n))
    Wi = cp.zeros(shape = (n,n))
    for i in range(n):
        for j in range(n):
            # i row, j column
            distsq = find_distsq(i,j)
            We[i,j] = cp.exp(- distsq/2/sigmaE) * (distsq <= r1sq)
        We[i] = we * We[i] / sum_exc
    for i in range(n):
        for j in range(n):
            distsq = find_distsq(i,j)
            Wi[i,j] = (distsq <= r2sq)
        Wi[i] = wi * Wi[i] / sum_inh

    return We, Wi

We, Wi = get_laplacian(neuron_shape[0]*neuron_shape[1], r1 = re, r2 = ri, wi=wi, we=we)


def perceive_to_get_stimulus(word_batch, codebook):
    stimulus = cp.dot(word_batch, codebook)  # word_batch = this_X = (256, 97), code_book = (97, 400)
    return stimulus   # shape: (256, 400)

def stimulate(stimulus):  # stimulus: (256, 20, 20)
    global exc_act
    global inh_act
    for t in range(int(max_act_fit)):
        exc_act_tm1 = cp.copy(exc_act)

        exc_input = cp.dot(exc_act, We)
        inh_input = cp.dot(inh_act, Wi)

        exc_act = exc_act + lr_act * (- leaky * exc_act + stimulus + exc_input - inh_input)
        inh_act = inh_act + lr_act * (- leaky * inh_act + exc_input)

        exc_act = cp.maximum(exc_act - threshold, 0) - cp.maximum(-exc_act - threshold, 0)
        inh_act = cp.maximum(inh_act - threshold, 0) - cp.maximum(-inh_act - threshold, 0)

        da = exc_act - exc_act_tm1
        relative_error = cp.sqrt(cp.square(da).sum()) / (eps + cp.sqrt(cp.square(exc_act_tm1).sum()))

    if relative_error < eps:
        return exc_act
    else:
        print("error = " + str(relative_error))
        print("exc_act = "+ str(exc_act))
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
#Phi = cp.load(get_fpath_from_configs() + "codebook.npy")

Phi = cp.load("codebook.npy")
#print("codebook = " +str(Phi))

emb_dim, num_units = Phi.shape
activity = cp.zeros(shape=(num_test_vocabs, num_units))


with open('../data/googleNgram/4vocabidx.pkl', 'rb') as f:
    vocabidx = pickle.load(f)

def plot_word_activations(words, filename=''):
    i = 0
    global bs
    bs = len(words)
    global exc_act
    global inh_act
    exc_act = cp.zeros(shape=(bs, neuron_shape[0]*neuron_shape[1]))  # shape should be (bs, neuron_shape)!
    inh_act = cp.zeros(shape=(bs, neuron_shape[0]*neuron_shape[1]))

    global activity
    word_batch, wp_idx = load_test_batch(words)
    #print("word_batch = " +str(word_batch))
    try:
        stimulus = perceive_to_get_stimulus(word_batch, Phi)
        activ = stimulate(stimulus)
        #activ = activ.reshape([bs, num_units])
        activity[wp_idx, :] = activ
    except RuntimeError as e:
        print(e)

    for word in words:
        try:
            activ = activity[vocabidx[word]]
            print("word '{}' = ".format(word)+str(activ))
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
    plot_word_activations(['universe', 'university', 'astronomy', 'college'], 'universe')
    plot_word_activations(['monarch', 'king', 'queen', 'female', 'prince', 'princess'], 'people')
    plot_word_activations(['cell', 'brain', 'organ', 'piano'], 'biology')
    #plot_word_activations(['brain'],'apple_test')

