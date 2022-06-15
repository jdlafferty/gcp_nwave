'''
==========
Date: June 15, 2022
Maintantainer:
Xinyi Zhong (xinyi.zhong@yale.edu)
Xinchen Du (xinchen.du@yale.edu)
Zhiyuan Long (zhiyuan.long@yale.edu)
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
    leaky: int = wi + we

@dataclass
class NDMConfig:
    pass


@dataclass
class ExperimentConfig:
    loader_name : str = "unigram97"
    ndm_name : str = "l1ActDoubleDecker"
    dl_name : str = "gradientDescent"
    input_dim : int = 97 
    neuron_shape : tuple = (40,40)
    gradient_steps : int = 30000
    batch_size : int = 256
        
    lr_act : float = 0.01
    lr_codebook: float = 0.01
    l1_target : float = 0.2
    threshold: float = 0.01


@dataclass
class Config:
    kernel :  KernelConfig = KernelConfig()
    ndm : NDMConfig =  NDMConfig()
    exp : ExperimentConfig = ExperimentConfig()
    srepr : str = MISSING


srepr = "test"
cfg = Config(srepr = srepr)