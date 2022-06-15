'''
==========
Date: Apr 7, 2022
Maintantainer: Xinyi Zhong (xinyi.zhong@yale.edu)
==========
This module implements dictionary learners.

The decisive attribute of DL is the dictionary it possesses.
Main actions: 
1) input batch data + dictionary => stimulus to neurons
2) neuron stimulus + input batch data => dictionary update
'''

from abc import ABC, abstractmethod 
import cupy as cp

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

    def __init__(self, input_dim, neuron_shape):
        # TODO, make them on GPU memory 
        # self.input_dim = cp.asarray(input_dim)
        self.neuron_shape = neuron_shape 
        self.codebook = cp.zeros(shape = (input_dim, *neuron_shape))

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
        (bs, input_dim) * (input_dim, *neuron_shape) -> (bs, *neuron_shape)
        '''
        # TODO ensure word_batch is a cp object
        stimulus = cp.tensordot(word_batch, self.codebook, axes=([1], [0]))
        return stimulus

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
        neuron_activation = cp.expand_dims(neuron_activation, axis = 1)
        # Shape (bs, input_dim)
        fitted_value = (self.codebook * neuron_activation).sum(axis = range(2,2+len(self.neuron_shape)))
        error = word_batch - fitted_value
        gradient = cp.tensordot(error, neuron_activation, axes=([0], [0]))
        self.codebook += self.lr_codebook * gradient

        # Return avg l0 norm, avg l1 norm and l2 loss 
        return cp.mean(neuron_activation > 0), cp.abs(neuron_activation).mean(), \
            cp.sqrt(cp.square(error).mean())




        



