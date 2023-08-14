from csv import DictReader
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy
import math
import argparse
import pickle

re = 5
ri = 3
wi = 5
we = 30
lr_act = 0.01
threshold = 0.01
leaky = wi + we
input_dim = 97
neuron_shape = (40, 40)

CPU = 1
if CPU:
    import numpy as cp
    from scipy.signal import convolve
else:
    import cupy as cp
    from cusignal.convolution.convolve import convolve

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

cnt = 0
word_embeddings = numpy.load('./' + 'data/googleNgram/embed100.npy')
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
exck, inhk = get_kernels(re = re, ri = ri, we = we, wi = wi)
print("exck = " +str(exck))
print("ink = " +str(inhk))

exck = cp.expand_dims(cp.asarray(exck), axis = 0)
inhk = cp.expand_dims(cp.asarray(inhk), axis = 0)
lr_act = cp.asarray(lr_act)
leaky = cp.asarray(leaky)
max_act_fit = cp.asarray(50)
threshold = threshold
eps = cp.asarray(5e-3)

def load_test_batch(words):
    idx = []
    for word in words:
        idx.append(vocabidx[word])
    word_batch = word_embeddings[idx, :]
    return word_batch, idx

def perceive_to_get_stimulus(word_batch, codebook):
    stimulus = cp.dot(word_batch, codebook).reshape((word_batch.shape[0], neuron_shape[0], neuron_shape[1]))  # word_batch = this_X = (256, 97), code_book = (97, 400)
    return stimulus   # shape: (256, 400)

def stimulate(stimulus):  # stimulus: (256, 20, 20)
    global exc_act
    global inh_act
    print(stimulus.shape)
    for t in range(int(max_act_fit)):
        exc_act_tm1 = cp.copy(exc_act)
        print(exc_act.shape)
        print(exck.shape)
        exc_input = convolve(exc_act, exck, mode="same")  # (256, 20, 20)
        inh_input = convolve(inh_act, inhk, mode="same")

        exc_act = exc_act + lr_act * (- leaky * exc_act + stimulus + exc_input - inh_input)
        inh_act = inh_act + lr_act * (- leaky * inh_act + exc_input)

        # Soft threshold
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

def plot_word_activations(words, filename=''):
    i = 0
    global bs
    bs = len(words)
    global exc_act
    global exc_act_dummy
    global inh_act
    global inh_act_dummy
    exc_act_dummy = cp.zeros(shape = (bs, neuron_shape[0]*neuron_shape[1] + 1))
    exc_act = cp.zeros(shape=(bs, neuron_shape[0] * neuron_shape[1]))  # shape should be (bs, neuron_shape)!
    inh_act_dummy = cp.zeros(shape = (bs, neuron_shape[0]*neuron_shape[1] + 1))
    inh_act = cp.zeros(shape=(bs, neuron_shape[0]*neuron_shape[1]))

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



Phi = cp.load("./" + "codebook.npy")

emb_dim, num_units = Phi.shape
activity = cp.zeros(shape=(num_test_vocabs, num_units))

with open('./' + 'data/googleNgram/4vocabidx.pkl', 'rb') as f:
    vocabidx = pickle.load(f)

if __name__ == "__main__":
    plot_word_activations(['brain'], 'test')


