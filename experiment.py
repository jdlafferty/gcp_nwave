'''
==========
Date: Apr 13
Maintantainer: Xinyi Zhong (xinyi.zhong@yale.edu)
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
from configs import cfg

from pallet import REGISTRY as loaderRegistry
Loader = loaderRegistry[cfg.exp.loader_name]

from nwave.dictlearner import REGISTRY as dlRegistry
DL = dlRegistry[cfg.exp.dl_name]

from nwave.neurondynamics import REGISTRY as ndmRegistry
NDM = ndmRegistry[cfg.exp.ndm_name]

from nwave.utils import get_kernels
k_exc, k_inh = get_kernels(re = cfg.kernel.re, ri = cfg.kernel.ri, \
    we = cfg.kernel.we, wi = cfg.kernel.wi)

#################################
# Training
#################################



#################################
# Evaluation
#################################
