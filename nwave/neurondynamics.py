'''
==========
Date: June 15, 2022
Maintantainer:
Xinyi Zhong (xinyi.zhong@yale.edu)
Xinchen Du (xinchen.du@yale.edu)
Zhiyuan Long (zhiyuan.long@yale.edu)
==========
This module implements wave dynamics on neurons 
The main attributes 
1) the organization of neurons in array whose values represent the activition level
2) any data structure to support defining the dynamic
The main method is the dynamic 
'''

from abc import ABC, abstractmethod 
import logging 
import pickle

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#import cupy as cp
#import cusignal
import numpy as cp
from scipy import signal

REGISTRY = {}

def register(cls_name):
    '''A decorator to set the name of the class and add the name to DataLoaders
    '''
    def registerer(cls):
        cls.__name__ = cls_name
        REGISTRY[cls_name] = cls
        return cls
    return registerer



class _NeuronDynamicsModel(ABC):
    '''Prototype for all neuron dynamics models 
    
    Methods
    -------
    stimulate : upon receiving the simulus, define how the neurons evolve
        Input : array (neuron shape) : stimulus
        Output : array (neuron shape): final neuron activation snapshot
    
    advance : advance neuron activition by one time step
        Input :  None
        Output : None
    
    evolve : change innate parameter of the neuron dynamic model
        Input : Dependends
        Output : new parameter 
    
    reset : ATNS
        Input : None
        Output : None 

    Attributes
    -------
    neuron_shape 
    '''
    def __init__(self) -> None:
        pass 

    @abstractmethod
    def reset(self):
        pass 

    @abstractmethod
    def stimulate(self, stimulus):
        return 
    
    @abstractmethod
    def advance(self):
        return 

    @abstractmethod
    def evolve(self):
        return 
    
    @abstractmethod
    def dump(self, fpath):
        return 
    
    @abstractmethod
    def load(self, fpath):
        return 
    


@register("l1ActDoubleDecker")
class L1ActDoubleDecker(_NeuronDynamicsModel):

    '''Activation and inhibition overlay with each other on the grid
    
    Methods
    -------
    
    Attributes
    -------
    exck : : activation kernel
    inhk : : inhibition kernel 
    lr_act : float : learn_rate for neuron activations
    exc_act
    inh_act
    '''
    
    def __init__(self, neuron_shape, leaky, exck, inhk, lr_act, l1_target, threshold, bs = 256) -> None:
        assert len(neuron_shape) == 2 

        self.neuron_shape = neuron_shape
        self.bs = bs
        self.exc_act = cp.zeros(shape = (bs, neuron_shape[0], neuron_shape[1]))   # shape should be (bs, neuron_shape)!
        self.inh_act = cp.zeros(shape = (bs, neuron_shape[0], neuron_shape[1]))

        self.exck = cp.expand_dims(cp.asarray(exck), axis = 0)
        self.inhk = cp.expand_dims(cp.asarray(inhk), axis = 0)
        self.lr_act = cp.asarray(lr_act)
        self.l1_target = cp.asarray(l1_target)
        self.leaky = cp.asarray(leaky)
        self.max_act_fit = cp.asarray(50)
        self.threshold = threshold
        self.eps = cp.asarray(5e-3)
    
    def reset(self) -> None:
        self.exc_act.fill(0)
        self.inh_act.fill(0)

    def stimulate(self, stimulus):  # stimulus: (256, 20, 20)
        for t in range(self.max_act_fit):
            self.advance(stimulus)

            da = self.exc_act - self.exc_act_tm1

            relative_error = cp.sqrt(cp.square(da).sum()) / (self.eps + cp.sqrt(cp.square(self.exc_act_tm1).sum()))

        if relative_error < self.eps:
            return self.exc_act
        else:
            logger.warning("Relative error end with {:.4f} and doesn't converge within the max fit steps".format(self.exc_act))
            return self.exc_act
    
    def advance(self, stimulus):

        self.exc_act_tm1 = cp.copy(self.exc_act)
        # exc_act : 256*20*20
        # exck : 1*7*7

        #exc_input = cusignal.convolve2d(self.exc_act, self.exck, mode = "same")
        #inh_input = cusignal.convolve2d(self.inh_act, self.inhk, mode = "same")
        exc_input = signal.convolve(self.exc_act, self.exck, mode="same")  # (256, 20, 20)
        inh_input = signal.convolve(self.inh_act, self.inhk, mode="same")

        self.exc_act , self.inh_act = self.exc_act + \
            self.lr_act * (
                - self.leaky * self.exc_act + stimulus + exc_input -inh_input
            ) , self.inh_act + \
                self.lr_act * (
                    -self.leaky * self.inh_act + exc_input
                )

        # Soft threshold 
        self.exc_act = cp.maximum(self.exc_act - self.threshold, 0) - cp.maximum(-self.exc_act - self.threshold, 0)
        self.inh_act = cp.maximum(self.inh_act - self.threshold, 0) - cp.maximum(-self.inh_act - self.threshold, 0)

    def evolve(self):
        dthreshold = cp.mean((cp.abs(self.exc_act) > 1e-4)) - self.l1_target
        self.threshold += .01 * dthreshold

    def dump(self, fpath):
        fname = fpath + "neurondynamics.pickle"
        with open(fname, 'wb') as f:
            pickle.dump([self.threshold], f)
    
    def load(self, fpath):
        fname = fpath + "neurondynamics.pickle"
        with open(fname, 'rb') as f:
            self.threshold = pickle.load(f)



    









