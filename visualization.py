'''
==========
Date: June 15, 2022
Maintainer:
Xinyi Zhong (xinyi.zhong@yale.edu)
Xinchen Du (xinchen.du@yale.edu)
Zhiyuan Long (zhiyuan.long@yale.edu)
==========
'''

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 6) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

import numpy as np 
from matplotlib.ticker import FuncFormatter


def vis_error(error, fpath, train_start_step = 0):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols = 3, figsize = (20,3))
    ax0.plot(error[:,0])
    ax0.set_title("reconstruction error")
    ax0.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%d" % (train_start_step+x)))
    ax1.plot(error[:,1])
    ax1.set_title("l1norm")
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%d" % (train_start_step+x)))
    ax2.plot(error[:,2])
    ax2.set_title("l0norm")
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%d" % (train_start_step+x)))
    
    fig.suptitle(fpath[10:].replace('/', ' ').strip())
    fig.savefig(fpath + 'errors{}ts.png'.format(train_start_step))
    plt.close()
