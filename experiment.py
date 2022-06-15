'''
==========
Date: June 15, 2022
Maintantainer:
Xinyi Zhong (xinyi.zhong@yale.edu)
Xinchen Du (xinchen.du@yale.edu)
Zhiyuan Long (zhiyuan.long@yale.edu)
==========
The experiment process
1. Input experiment set up and model parameters
2. Training 
3. Evaluation 
Visualization on evaluation is in an seperate file
'''


#################################
# Experiment Setup 
#################################

#import cupy as cp
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

fpath = "result"

#################################
# Training
#################################
loader = Loader(cfg.exp.batch_size)
dl = DL(input_dim = cfg.exp.input_dim, neuron_shape = cfg.exp.neuron_shape, lr_codebook = cfg.exp.lr_codebook)
ndm = NDM(neuron_shape = cfg.exp.neuron_shape, leaky = cfg.kernel.leaky, exck = k_exc, inhk = k_inh,
         lr_act = cfg.exp.lr_act, l1_target = cfg.exp.l1_target, threshold = cfg.exp.threshold)

wave = wave.Wave(dl, ndm)

l2_loss, l1_loss, l0_loss = wave.train_through_loader(loader = loader, gradient_steps=cfg.exp.gradient_steps)

errors = cp.column_stack((l2_loss, l1_loss, l0_loss))
vis_error(errors, fpath)



#################################
# Evaluation
#################################

# in other py files