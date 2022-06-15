'''
==========
Date: Thu Apr 14
Maintantainer: Xinyi Zhong (xinyi.zhong@yale.edu)
==========
Config class and CLI to set config
'''
from dataclasses import dataclass

from omegaconf import OmegaConf, MISSING

@dataclass
class KernelConfig:
    ri : int = 5
    re : int = 3
    wi : int = 5
    we : int = 30

@dataclass
class NDMConfig:
    a: int = 1 
    b: int = 1


@dataclass
class ExperimentConfig:
    loader_name : str = "unigram97"
    ndm_name : str = "l1ActDoubleDecker"
    dl_name : str = "gradientDescent"
    input_dim : int = 97 
    neuron_shape : tuple = (20,20)
    train_steps : int = 1000
    batch_size : int = 256


@dataclass
class Config:
    kernel :  KernelConfig = KernelConfig()
    ndm : NDMConfig =  NDMConfig()
    exp : ExperimentConfig = ExperimentConfig()
    srepr : str = MISSING

srepr = "test"
cfg = OmegaConf.structured(Config(srepr=srepr))
cfg.kernel.we = 3
print(OmegaConf.to_yaml(cfg))


