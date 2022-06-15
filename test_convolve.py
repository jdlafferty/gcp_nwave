import cupy as cp
import numpy as np
from scipy import signal
from cusignal.convolution.convolve import convolve, fftconvolve
import timeit

side_len = 100
neuron_shape = (256, side_len, side_len)
num_samples = int(1e2)

# Generate Data on CPU
exc_decay = np.ones(neuron_shape)
exc_rise = np.ones(neuron_shape)
k_exc_3d = np.random.rand(1,9,9)

# run on CPU
t_cpu = timeit.timeit(lambda: signal.convolve(exc_decay - exc_rise, k_exc_3d, mode='same'), number = num_samples)
print(f"CPU: {t_cpu}")

# run on GPU
t_cpu_gpu = timeit.timeit(lambda: convolve(cp.asarray(exc_decay-exc_rise), cp.asarray(k_exc_3d), mode='same'), number = num_samples)
print(f"CPU to GPU: {t_cpu_gpu}")

# Generate Data on GPU
g_exc_decay = cp.ones(neuron_shape)
g_exc_rise = cp.ones(neuron_shape)
g_k_exc_3d = cp.random.rand(1,9,9)

t_gpu = timeit.timeit(lambda: convolve(g_exc_decay-g_exc_rise, g_k_exc_3d, mode='same'), number = num_samples)
print(f"GPU: {t_gpu}")

print("GPU over CPU: {:d}x".format(int(t_cpu/t_gpu)))
print("GPU over CPU->GPU: {:d}x".format(int(t_cpu_gpu/t_gpu)))