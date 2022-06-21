'''
==========
Date: June 20, 2022
Maintainer:
Xinyi Zhong (xinyi.zhong@yale.edu)
Xinchen Du (xinchen.du@yale.edu)
Zhiyuan Long (zhiyuan.long@yale.edu)
==========
Config class and CLI to set config
'''
from dataclasses import dataclass

from omegaconf import OmegaConf, MISSING

from csv import DictReader

import math

params = []
with open('parameter.csv', 'r') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
    csv_dict_reader = DictReader(read_obj)
    # iterate over each line as a ordered dictionary
    for row in csv_dict_reader:
        # row variable is a dictionary that represents a row in csv
        params.append(row)

# trial 1, only one row of params
param = params[0]

def get_neuron_shape(x):
    y = int(math.sqrt(x))
    return (y, y)


@dataclass
class KernelConfig:
    ri : int = int(param['ri'])
    re : int = int(param['re'])
    wi : int = int(param['wi'])
    we : int = int(param['we'])
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
    neuron_shape : tuple = get_neuron_shape(int(param['neuron_shape']))
    gradient_steps : int = int(param['gradient_steps'])
    batch_size : int = int(param['batch_size'])
        
    lr_act : float = float(param['lr_act'])
    lr_codebook: float = float(param['lr_codebook'])
    l0_target : float = float(param['l0_target'])
    threshold: float = float(param['threshold'])


@dataclass
class Config:
    kernel :  KernelConfig = KernelConfig()
    ndm : NDMConfig =  NDMConfig()
    exp : ExperimentConfig = ExperimentConfig()
    srepr : str = MISSING


srepr = "test"
cfg = Config(srepr = srepr)