'''
==========
Date: June 15, 2022
Maintainer:
Xinyi Zhong (xinyi.zhong@yale.edu)
Xinchen Du (xinchen.du@yale.edu)
Zhiyuan Long (zhiyuan.long@yale.edu)
==========
Encapsulate dictlearner and neurodynamics to form a wave object
'''

from configs import *
if get_argsprocessor() == "CPU":
    import numpy as cp
elif get_argsprocessor() == "GPU":
    import cupy as cp

from tqdm import trange

from configs import cfg
from nwave.utils import get_fpath_from_configs


class Wave():
    '''
    
    Methods
    -------
    train_dictionary : 
    get_excitation : 
    
    Attributes
    -------
    dict_learner : DictLearner 
    neuron_dynmic_model : NeuronDynamicsModel
    '''

    def __init__(self, dict_learner, neuron_dynamic_model) -> None:
        self.dict_learner = dict_learner
        self.neuron_dynamic_model = neuron_dynamic_model


    def train_dictionary(self, word_batch):  # word_batch : 256 * 97
        # Get stimulus
        stimulus = self.dict_learner.perceive_to_get_stimulus(word_batch)  # stimulus: (256, 20， 20)
        # Get neuron activity
        activation = self.neuron_dynamic_model.stimulate(stimulus)
        # Neuron model evolve and reset
        self.neuron_dynamic_model.evolve()
        # Update codebook
        l0l, l1l, l2l = self.dict_learner.update_codebook(word_batch, activation)
        self.neuron_dynamic_model.reset()
        return l0l, l1l, l2l

    def save(self, fpath):
        self.neuron_dynamic_model.dump(fpath)
        self.dict_learner.dump(fpath)
    
    def load(self, fpath):
        self.neuron_dynamic_model.load(fpath)
        self.dict_learner.load(fpath)

    def train_through_loader(self, loader, gradient_steps=5000, initial_step=0):
        l2_loss = []
        l1_loss = []
        l0_loss = []
        tbar = trange(initial_step, initial_step + gradient_steps, desc='Training', leave=True, miniters=100)

        for i in tbar:
            this_X, _ = loader.load_train_batch()  #this_X : 256 * 97

            l0l, l1l, l2l = self.train_dictionary(word_batch = this_X) #this_X : 256 * 97

            l2_loss.append(l2l)
            l1_loss.append(l1l)
            l0_loss.append(l0l)

            if i % 100 == 0:
                tbar.set_description("loss=%.3f sparsity=%2.2f%% lmda=%.3f" % \
                                     (l2l, 100 * l0l, self.neuron_dynamic_model.threshold))
                tbar.refresh()

        self.dict_learner.dump(get_fpath_from_configs(cfg))  # fpath

        return l2_loss, l1_loss, l0_loss
