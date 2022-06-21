'''
==========
Date: June 15, 2022
Maintainer:
Xinyi Zhong (xinyi.zhong@yale.edu)
Xinchen Du (xinchen.du@yale.edu)
Zhiyuan Long (zhiyuan.long@yale.edu)
==========
This module implements dictionary learners.

The decisive attribute of DL is the dictionary it possesses.
Main actions: 
1) input batch data + dictionary => stimulus to neurons
2) neuron stimulus + input batch data => dictionary update
'''

from abc import ABC, abstractmethod
#import cupy as cp
import numpy as cp
from tqdm import trange

REGISTRY = {}

def register(cls_name):
    '''A decorator to set the name of the class and add the name to DataLoaders
    '''
    def registerer(cls):
        cls.__name__ = cls_name
        REGISTRY[cls_name] = cls
        return cls
    return registerer


class _DictionaryLearner(ABC):
    '''Prototype for all dictionary learner
    
    Methods
    -------
    perceive_to_get_stimulus: receive input data and create stimulus on neurons
        Input : array (bs, input_dim) : training batch 
        Output : array (neuron_shape): stimulus on neurons 

    update_codebook: update the codebook/dictionary
        Input : array (bs, input_dim) : traininng batch 
        Input : array (neuron_shape) : neuron final activity
        Output : tuple : errors/metrics to monitor

    Attributes
    -------
    input_dim: int
    neuron_dynamics_model: NeuronDynamicsModel
    neuron_shape : tuple of int
    codebook : array @ (input_dim, *neuron_shape)
    '''

    def __init__(self, input_dim, neuron_shape):   # input_dim = 97
        self.input_dim = cp.asarray(input_dim)
        self.neuron_shape = neuron_shape
        self.codebook = 0.3 * cp.random.rand(input_dim, neuron_shape[0]*neuron_shape[1])  # 97 * 400

    @abstractmethod
    def perceive_to_get_stimulus(self, word_batch):
        pass 

    @abstractmethod
    def update_codebook(self, word_batch, neuron_activation):
        pass 

    def dump(self, fpath):
        fname = fpath + "codebook.npy"
        cp.save(fname, self.codebook)
    
    def load(self, fpath):
        fname = fpath + "codebook.npy"
        self.codebook = cp.load(fname)

        
@register("gradientDescent")
class GradientDescent(_DictionaryLearner):
    '''Gradient Descent based on reconstruction error
    
    Methods
    -------
    
    Attributes
    -------
    input_dim: int
    neuron_dynamics_model: NeuronDynamicsModel
    neuron_shape : tuple of int
    codebook : array @ (input_dim, *neuron_shape)
    lr_codebook : code book learning rate 
    '''
    
    def __init__(self, input_dim, neuron_shape, lr_codebook):
        super().__init__(input_dim, neuron_shape)
        self.lr_codebook = cp.asarray(lr_codebook)

    
    def perceive_to_get_stimulus(self, word_batch):
        '''
        (bs, input_dim) * (input_dim, *neuron_shape) -> (bs, *neuron_shape)   bs = 256
        '''
        stimulus = cp.dot(word_batch, self.codebook).reshape((word_batch.shape[0], self.neuron_shape[0], self.neuron_shape[1]))  # word_batch = this_X = (256, 97), code_book = (97, 400)

        return stimulus   # shape: (256, 400)

    def update_codebook(self, word_batch, neuron_activation):
        '''
        
        Parameters
        -------
        word_batch : array (bs, input_dim)
        neuron_action : array (input_dim, *neuron_shape)
        
        Returns
        -------
        '''
        # Shape (bs, input_dim, *neuron_shape)
        bs = cp.shape(neuron_activation)[0]
        neuron_activation = neuron_activation.reshape(bs, self.neuron_shape[0]*self.neuron_shape[1])
        # shape of code_book: 97*400
        # shape of neuron activation: 256*400
        # shape of word_batch: 256*97
        # Shape (bs, input_dim)

        fitted_value = cp.dot(neuron_activation, cp.transpose(self.codebook))

        error = word_batch - fitted_value
        gradient = cp.dot(cp.transpose(error), neuron_activation)
        self.codebook += self.lr_codebook * (gradient - cp.vstack(cp.mean(gradient, axis=1)))
        self.codebook = self.codebook / cp.maximum(cp.sqrt(cp.square(self.codebook).sum(axis=0)), 1e-8)

        return cp.mean(cp.abs(neuron_activation) > 1e-4), cp.abs(neuron_activation).mean(), \
            cp.sqrt(cp.square(error).sum())
    


