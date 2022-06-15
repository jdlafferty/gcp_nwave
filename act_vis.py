'''
==========
Date: June 15, 2022
Maintantainer:
Xinyi Zhong (xinyi.zhong@yale.edu)
Xinchen Du (xinchen.du@yale.edu)
Zhiyuan Long (zhiyuan.long@yale.edu)
==========
'''

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 6)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

#################################
# Experiment Setup
#################################

import numpy as cp
from configs import cfg
from visualization import vis_error

import nwave.wave as wave

from dataloader import REGISTRY as loaderRegistry
Loader = loaderRegistry[cfg.exp.loader_name]

from nwave.dictlearner import REGISTRY as dlRegistry
DL = dlRegistry[cfg.exp.dl_name]

from nwave.neurondynamics import REGISTRY as ndmRegistry
NDM = ndmRegistry[cfg.exp.ndm_name]

from nwave.utils import get_kernels
k_exc, k_inh = get_kernels(re = cfg.kernel.re, ri = cfg.kernel.ri, \
   we = cfg.kernel.we, wi = cfg.kernel.wi)

fpath = "result/"

if __name__ == "__main__":

    Phi = cp.load(fpath + "codebook.npy")

    emb_dim, num_units = Phi.shape  # 97 * 400

    loader = Loader(cfg.exp.batch_size)
    dl = DL(input_dim = cfg.exp.input_dim, neuron_shape = cfg.exp.neuron_shape, lr_codebook = cfg.exp.lr_codebook)
    ndm = NDM(neuron_shape = cfg.exp.neuron_shape, leaky = cfg.kernel.leaky, exck = k_exc, inhk = k_inh,
         lr_act = cfg.exp.lr_act, l1_target = cfg.exp.l1_target, threshold = cfg.exp.threshold)

    loader.cnt = 0
    activity = cp.zeros(shape=(loader.num_test_vocabs, num_units))  # loader.num_test_vocabs = 20000

    dl.codebook = Phi

    for t in range((loader.num_test_vocabs - 1) // loader.batch_size + 1):  # batch_size = 128
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

