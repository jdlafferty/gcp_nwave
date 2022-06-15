'''
==========
Date: Apr 7, 2020
Maintantainer: Xinyi Zhong (xinyi.zhong@yale.edu)
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


import cupy as cp
import cusignal 

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
    
    def __init__(self, neuron_shape, leaky, exck, inhk, lr_act, l1_target) -> None:
        assert len(neuron_shape) == 2 
        self.exc_act = cp.zeros(*neuron_shape)
        self.inh_act = cp.zeros(*neuron_shape)
        self.exck = cp.asarray(exck)
        self.inhk = cp.asarray(inhk)
        self.lr_act = cp.asarray(lr_act)
        self.l1_target = cp.asarray(l1_target)
        self.leaky = cp.asarray(leaky)
        self.max_act_fit = cp.asarry(50)
        self.threshold = 0 # TODO what is the initial threshould? 
        self.eps = cp.asarray(5e-3)
    
    def reset(self) -> None:
        self.exc_act.fill(0)
        self.inh_act.fill(0)

    def stimulate(self, stimulus):

        exc_acttm1 = self.exc_act
        
        for t in range(self.max_act_fit): 
            exc_act = self.advance(stimulus)

            da = exc_act - exc_acttm1
            relative_error = cp.square(da).mean() / cp.square(exc_acttm1).mean()

            if relative_error < self.eps:
                return exc_act
            else:
                logger.warning("Relative error end with {:.4f} and doesn't converge within the max fit steps".format(exc_act))
                return exc_act
    
    def advance(self, stimulus):
        exc_input = cusignal.convolve2d(self.exc_act, self.exck, mode = "same")
        inh_input = cusignal.convolve2d(self.inh_act, self.inhk, mode = "same")

        # Python evaluates RHS of assignment first 
        self.exc_act , self.inh_act = self.exc_act + \
            self.lr_act * (
                - self.leaky * self.exc_act + stimulus + exc_input -inh_input
            ) , self.inh + \
                self.lr_act * (
                    -self.leaky * self.inh_act + exc_input
                )
        # Soft threshold 
        self.exc_act = cp.maximum(self.exc_act - self.threshold, 0) - cp.maximum(-self.exc_act - self.threshold, 0)
        self.inh_act = cp.maximum(self.inh_act - self.threshold, 0) - cp.maximum(-self.inh_act - self.threshold, 0)

    def evolve(self):
        dthreshold = (self.exc_act > 0).mean() - self.l1_target 
        self.threshold += .01 * dthreshold

    def dump(self, fpath):
        fname = fpath + "neurondynamics.pickle"
        with open(fname, 'wb') as f:
            pickle.dump([self.threshold], f)
    
    def load(self, fpath):
        fname = fpath + "neurondynamics.pickle"
        with open(fname, 'rb') as f:
            self.threshold = pickle.load(f)



    









