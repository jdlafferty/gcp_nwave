'''
==========
Date : Apr 7, 2022 
Maintainer: Xinyi Zhong (xinyi.zhong@yale.edu)
==========
Uniform interface to load data
'''

REGISTRY = {}
import numpy

def register(cls_name):
    '''A decorator to set the name of the class and add the name to DataLoaders
    '''
    def registerer(cls):
        cls.__name__ = cls_name
        REGISTRY[cls_name] = cls
        return cls
    return registerer

from abc import ABC, abstractmethod 
from configs import *

if get_argsprocessor() == "CPU":
    import numpy as cp
elif get_argsprocessor() == "GPU":
    import cupy as cp

class _DataLoader(ABC): 
    '''Universal interface for all data loaders
    
    Methods
    -------
    load_train_batch: random subsample
        Input: int: batch_size
        Output: tuple: (wrod_batch, idx), where word_batch @ array (bs, input_dim)
    
    load_test_batch: load word in batch sequentially
        Input: int: batch_size 
        Output: tuple: (word_batch, idx), where word_batch @ array (bs, input_dim)

    
    Attributes
    -------

    '''
    def __init__(self):
        pass 
        
    @abstractmethod
    def load_train_batch(self):
        return
    
    @abstractmethod
    def load_test_batch(self):
        return 

@register("unigram97")
class UnigramLoader(_DataLoader): 

    def __init__(self, batch_size):
        self.batch_size = batch_size    
        self.cnt = 0
        self.word_embeddings = numpy.load('../data/googleNgram/embed100.npy')
        self.word_embeddings = numpy.delete(self.word_embeddings, [55, 58, 84], axis = 1)
        self.word_embeddings = cp.asarray(self.word_embeddings)
        self.word_freq = cp.load("../data/googleNgram/1gramSortedFreq.npy")
        self.num_train_vocabs = self.word_freq.shape[0]
        self.num_test_vocabs = numpy.asarray(20000)
        self.SUBSAMPLE_SIZE = numpy.asarray(4096)

    def __str__(self):
        return "Google 1 Gram freq; Glove Embedding with dim=97 after removing 3 systematic bias"

    def sample_word_idx(self, batch_size):
        subsampled_idx = numpy.random.randint(0, self.num_train_vocabs, self.SUBSAMPLE_SIZE)
        subsampled_idx = cp.asarray(subsampled_idx)
        prob = self.word_freq[subsampled_idx]
        prob = prob / cp.abs(prob).sum()
        sampled_locs = cp.random.choice(a = subsampled_idx, size = batch_size, replace = False, p=prob)
        return sampled_locs

    def load_train_batch(self):
        sampled_idx = self.sample_word_idx(self.batch_size)
        word_batch = self.word_embeddings[sampled_idx,:]
        return word_batch, sampled_idx

    def load_test_batch(self):
        if self.cnt > self.num_test_vocabs - self.batch_size:
            self.cnt=self.num_test_vocabs - self.batch_size
        idx = cp.arange(self.cnt, self.cnt + self.batch_size)
        self.cnt += self.batch_size
        word_batch = self.word_embeddings[idx,:]
        return word_batch, idx
    
    def load_by_idx(self, idx):
        word_batch = self.word_embeddings[idx,:]
        return word_batch, idx
