'''
==========
Date: July 5, 2022
Maintainer:
Xinyi Zhong (xinyi.zhong@yale.edu)
Xinchen Du (xinchen.du@yale.edu)
Zhiyuan Long (zhiyuan.long@yale.edu)
==========
The experiment process
1. Input experiment set up and model parameters
2. Train
3. Save and plot receptive fields
4. Save and plot activations
'''

#################################
# Experiment Setup 
#################################

from configs import *
import numpy
if get_argsprocessor() == "GPU":
    import cupy as cp
elif get_argsprocessor() == "CPU":
    import numpy as cp

import time

from nwave.utils import *
import nwave.wave as wave
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv

from dataloader import REGISTRY as loaderRegistry
Loader = loaderRegistry[cfg.exp.loader_name]

from nwave.dictlearner import REGISTRY as dlRegistry
DL = dlRegistry[cfg.exp.dl_name]

from nwave.neurondynamics import REGISTRY as ndmRegistry
NDM = ndmRegistry[cfg.exp.ndm_name]


k_exc, k_inh = get_kernels(re = cfg.kernel.re, ri = cfg.kernel.ri, \
   we = cfg.kernel.we, wi = cfg.kernel.wi)

fpath = get_fpath_from_configs(cfg)
mymkdir(fpath)

#################################
# Training
#################################
loader = Loader(cfg.exp.batch_size)
dl = DL(input_dim = cfg.exp.input_dim, neuron_shape = cfg.exp.neuron_shape, lr_codebook = cfg.exp.lr_codebook)
ndm = NDM(neuron_shape = cfg.exp.neuron_shape, leaky = cfg.kernel.leaky, exck = k_exc, inhk = k_inh,
         lr_act = cfg.exp.lr_act, l0_target = cfg.exp.l0_target, threshold = cfg.exp.threshold)

wave = wave.Wave(dl, ndm)

start = time.time()
l2_loss, l1_loss, l0_loss = wave.train_through_loader(loader = loader, gradient_steps=cfg.exp.gradient_steps)
end = time.time()

seconds = end - start
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
time = str(int(h))+"h "+str(int(m))+"m "+str(int(s))+"s"
print("The training time is: " + str(time))

errors = numpy.column_stack((l2_loss, l1_loss, l0_loss))
vis_error(errors, fpath)

with open('parameter.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [i for i in reader]

if get_argsprocessor() == 'CPU':
    rows[get_argsrow()][11] = time
    rows[get_argsrow()][13] = round(l2_loss[-1], 2)
elif get_argsprocessor() == 'GPU':
    rows[get_argsrow()][12] = time
    rows[get_argsrow()][14] = round(l2_loss[-1], 2)


with open('parameter.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(rows)):
        writer.writerow(rows[i])


#################################
# Save and plot receptive field
#################################
fpath = get_fpath_from_configs(cfg)

Phi = cp.load(fpath + "codebook.npy")

emb_dim, num_units = Phi.shape  # 97 * 400

ndm_rc = NDM(neuron_shape = cfg.exp.neuron_shape, leaky = cfg.kernel.leaky, exck = k_exc, inhk = k_inh,
         lr_act = cfg.exp.lr_act, l0_target = cfg.exp.l0_target, threshold = cfg.exp.threshold, bs = 97)

batch = cp.eye(emb_dim)
batch = batch - cp.mean(batch, axis=1)
batch = batch / cp.std(batch, axis=1)

dl.codebook = Phi
stimulus = dl.perceive_to_get_stimulus(batch)
RC = ndm_rc.stimulate(stimulus)
RC = RC.reshape([emb_dim, num_units])

cp.save(fpath + "receptive_fields.npy", RC)

plot_PCA(RC, fpath + 'RC.pdf')

#################################
# Save activations
#################################
loader.cnt = 0
activity = cp.zeros(shape=(loader.num_test_vocabs, num_units))  # loader.num_test_vocabs = 20000

for t in range((loader.num_test_vocabs - 1) // loader.batch_size + 1):  # batch_size = 256
    word_batch, wp_idx = loader.load_test_batch()
    try:
        stimulus = dl.perceive_to_get_stimulus(word_batch)
        activ = ndm.stimulate(stimulus)
        activ = activ.reshape([256, num_units])
        activity[wp_idx, :] = activ
    except RuntimeError as e:
        print(e)

with open(fpath + '/uactivity.npy', 'wb') as f:
    cp.save(f, activity)

#################################
# Plot activations
#################################
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
            im = ax.imshow(activ.reshape(cfg.exp.neuron_shape[0], cfg.exp.neuron_shape[1]),
                               cmap='jet', interpolation='gaussian', vmin=-l0norm, vmax=l0norm)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.2)
            plt.colorbar(im, cax=cax)
            ax.set_title("{}".format(word), fontsize=24)
            ax.set_axis_off()
            if len(filename) > 0:
                plt.savefig(fpath + '%s_%d.pdf' % (filename, i))
                i = i + 1


plot_word_activations(['technology', 'microsoft', 'intel', 'ibm', 'apple', 'banana'], 'tech')
plot_word_activations(['universe','university', 'astronomy', 'college'], 'universe')
plot_word_activations(['monarch', 'king', 'queen', 'female', 'prince', 'princess'], 'people')
plot_word_activations(['cell', 'brain', 'organ', 'piano'], 'biology')


