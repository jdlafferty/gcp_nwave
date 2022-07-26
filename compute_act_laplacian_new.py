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
side_length = neuron_shape[0]
gradient_steps = int(param['gradient_steps'])

lr_act = float(param['lr_act'])
print("lr_act = " +str(lr_act))
lr_codebook = float(param['lr_codebook'])
l0_target = float(param['l0_target'])
threshold = float(param['threshold'])
print("threshold = " +str(threshold))

sigmaE = 3

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
    return "../result/row{}_compute_act_laplacian_new/".format(get_argsrow())

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

#######################
# parallel algorithm
#######################

#####################################
# Compute num_E_nbs and num_I_nbs
#####################################

def get_num_nbs(r):
    count = 0
    for i in range((r+1)**2):
        xi = i // (r+1)
        yi = i % (r+1)
        distsq = xi**2 + yi**2
        if distsq <= r**2:
            count+=1

    num_nbs = (count-(r+1))*4 +1
    return num_nbs

num_E_nbs = get_num_nbs(re)  # 29

num_I_nbs = get_num_nbs(ri)  # 81


#print("num_E_nbs = " + str(num_E_nbs))
#print("num_I_nbs = " +str(num_I_nbs))

#####################################
# compute index set of each neurons
#####################################

def get_struct(r):
    rsq = r**2
    l = cp.zeros(2*r+1)
    for i in range(2*r+1):
        count = 0
        for j in range(2*r+1):
            distsq = (i - r) ** 2 + (j - r) ** 2
            if distsq <= rsq:
                count += 1
        l[i] = count
    return l

def get_index_from_position(xi, yi, xj, yj, r):
    r = int(r)
    xi = int(xi)
    yi = int(yi)
    xj = int(xj)
    yj = int(yj)
    l = get_struct(r)
    core_index = (get_num_nbs(r)-1)/2
    if yi == yj:
        index = core_index + (xj - xi)
        index = int(index)
        return index
    else:
        diff = 0
        for i in range(int(abs(yj - yi)-1)):
            diff += l[r-i-1]
        if (yi > yj):
            index = core_index - (l[r]-1)/2 - diff - (l[r-(yi - yj)]+1)/2 + (xj - xi)
            index = int(index)
            return index
        elif (yi < yj):
            index = core_index + (l[r]-1)/2 + diff + (l[r-(yi - yj)]+1)/2 + (xj - xi)
            index = int(index)
            return index


def compute_indexset(r, num_nbs):
    set = cp.zeros(shape=(neuron_shape[0]*neuron_shape[1], num_nbs))
    set = set.astype(int)
    set.fill(neuron_shape[0]*neuron_shape[1])
    #print(set)
    for i in range(neuron_shape[0]*neuron_shape[1]):
        xi = i // side_length
        yi = i % side_length
        for j in range(neuron_shape[0]*neuron_shape[1]):
            xj = j//side_length
            yj = j % side_length
            distsq = (xi - xj)**2 + (yi - yj)**2
            if distsq <= r**2:
                index = get_index_from_position(xi, yi, xj, yj, r)
                index = int(index)
                #print("index = " + str(index))
                set[i][int(index)] = j
                #print(set[i][index])
    return set

N_E = compute_indexset(re, num_E_nbs)
N_I = compute_indexset(ri, num_I_nbs)

#print("N_E = " +str(N_E))
# print("N_I = " +str(N_I))

#####################################
# compute weight kernels W_E and W_I
#####################################
W_E = cp.zeros(num_E_nbs)
W_I = cp.zeros(num_I_nbs)

count_E = 0
for i in range((2*re+1)**2):
    xi = i // (2*re+1)
    yi = i % (2*re+1)
    distsq = (xi - re)**2 + (yi - re)**2
    if distsq <= re**2:
        W_E[count_E] = cp.exp(- distsq/2/sigmaE)
        count_E += 1

#print("count_E = " + str(count_E))
W_E = we * W_E / cp.sum(W_E)
#print("W_E = " +str(W_E))

count_I = 0
for i in range((2*ri+1)**2):
    xi = i // (2 * ri + 1)
    yi = i % (2 * ri + 1)
    distsq = (xi - ri) ** 2 + (yi - ri) ** 2
    if distsq <= ri**2:
        W_I[count_I] = 1
        count_I += 1

#print("count_I = " + str(count_I))
W_I = wi * W_I / cp.sum(W_I)
#print("W_I = " +str(W_I))

###########################
# Update algorithms
###########################

def activation_update(a):
    b = cp.zeros(shape=(bs, neuron_shape[0] * neuron_shape[1]))
    for k in range(bs):
        r = cp.zeros(neuron_shape[0]*neuron_shape[1])
        for i in range(neuron_shape[0]*neuron_shape[1]):
            r[i] = - leaky * a[k][i]
            for j in range(num_E_nbs):
                r[i] += W_E[j] * a[k][N_E[i][j]]
            for j in range(num_I_nbs):
                r[i] -= W_I[j] * a[k][N_I[i][j]]

        for i in range(neuron_shape[0]*neuron_shape[1]):
            a[k][i] = r[i]

    for k in range(bs):
        for n in range(neuron_shape[0]*neuron_shape[1]):
            b[k][n] = a[k][n]
    return b


def perceive_to_get_stimulus(word_batch, codebook):
    stimulus = cp.dot(word_batch, codebook)  # word_batch = this_X = (256, 97), code_book = (97, 400)
    return stimulus   # shape: (256, 400)

def stimulate(stimulus):  # stimulus: (256, 20, 20)
    global exc_act
    global activation_dummy
    for t in range(int(max_act_fit)):
        exc_act_tm1 = cp.copy(exc_act)
        #print("#######################################\n")
        #print("activation_dummy0 = " +str(activation_dummy))
        #print("exc_act0 = " +str(exc_act))

        delta_a = activation_update(activation_dummy)
        #print("activation_dummy1 = " + str(activation_dummy))
        #print("delta_a = " +str(delta_a))

        exc_act = exc_act + lr_act * (cp.asarray(stimulus) + cp.asarray(delta_a))  # dimension problem

        exc_act = cp.maximum(exc_act - threshold, 0) - cp.maximum(-exc_act - threshold, 0)

        #print("exc_act1 = " + str(exc_act))
        #print("#######################################\n")
        for i in range(bs):
            for j in range(neuron_shape[0]*neuron_shape[1]):
                activation_dummy[i][j] = exc_act[i][j]

        da = exc_act - exc_act_tm1
        relative_error = cp.sqrt(cp.square(da).sum()) / (eps + cp.sqrt(cp.square(exc_act_tm1).sum()))

    if relative_error < eps:
        return exc_act
    else:
        print("error = " + str(relative_error))
        print("exc_act = " + str(exc_act))
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

Phi = cp.load("codebook.npy")

emb_dim, num_units = Phi.shape
activity = cp.zeros(shape=(num_test_vocabs, num_units))


with open('../data/googleNgram/4vocabidx.pkl', 'rb') as f:
    vocabidx = pickle.load(f)

def plot_word_activations(words, filename=''):
    i = 0
    global bs
    bs = len(words)
    global exc_act
    global activation_dummy
    global inh_act
    activation_dummy = cp.zeros(shape = (bs, neuron_shape[0]*neuron_shape[1] + 1))
    exc_act = cp.zeros(shape=(bs, neuron_shape[0] * neuron_shape[1]))  # shape should be (bs, neuron_shape)!
    #inh_act = cp.zeros(shape=(bs, neuron_shape[0]*neuron_shape[1]))

    global activity
    word_batch, wp_idx = load_test_batch(words)
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
            print("word '{}' = ".format(word) + str(activ))
        except Exception:
            print("word: {} not found".format(word))
        else:
            #activ = cp.delete(activ, [1600], axis = 1)  # delete the last dummy index
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
            #plt.show()



if __name__ == "__main__":
    plot_word_activations(['technology', 'microsoft', 'intel', 'ibm', 'apple', 'banana'], 'tech')
    plot_word_activations(['universe', 'university', 'astronomy', 'college'], 'universe')
    plot_word_activations(['monarch', 'king', 'queen', 'female', 'prince', 'princess'], 'people')
    plot_word_activations(['cell', 'brain', 'organ', 'piano'], 'biology')
    #plot_word_activations(['apple'],'apple_test')


