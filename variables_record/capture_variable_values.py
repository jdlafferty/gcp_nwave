import numpy
import numpy as np
# import cupy as np
# from cusignal.convolution.convolve import convolve
from scipy.signal import convolve
from tqdm import trange



# store the floating points
f = []
# store the array and matrix
m = []



ri = 5
re = 3
wi = 5
we = 30
leaky = wi + we
bs = 256
imbed_dim = 97
neuron_shape = (40, 40)
lr_act = 0.01
lr_Phi = 0.01
l0_target = 0.1
threshold = 0.01
initial_step = 0
gradient_steps = 2500
sigmaE = 3
max_act_fit = 50
eps = 5e-3

f.extend([ri, re, wi, we, leaky, bs, imbed_dim, neuron_shape[0]*neuron_shape[1], lr_act, lr_Phi, l0_target, threshold, initial_step, gradient_steps, sigmaE, max_act_fit, eps])

num_train_vocabs = 55529

f.append(num_train_vocabs)

def load_train_batch():
    sampled_idx = np.random.choice(num_train_vocabs, bs, replace=False)
    m.append(np.copy(sampled_idx))
    word_batch = word_embeddings[sampled_idx,:]
    # m.append(np.copy(word_batch))
    return word_batch


def get_kernels(re, ri, wi=5, we=30, sigmaE = 3):
    k_exc = np.zeros([2*re+1, 2*re+1])
    k_inh = np.zeros([2*ri+1, 2*ri+1])
    for i in range(2*re+1):
        for j in range(2*re+1):
            # i row, j column
            distsq = (i-re)**2+(j-re)**2
            f.append(distsq)
            k_exc[i,j] = np.exp(- distsq/2/sigmaE) * (distsq <= re**2)

    m.append(np.copy(k_exc))
    k_exc = we * k_exc / np.sum(k_exc)
    # m.append(np.copy(k_exc))

    for i in range(2*ri+1):
        for j in range(2*ri+1):
            # i row, j column
            distsq = (i-ri)**2+(j-ri)**2
            f.append(distsq)
            k_inh[i,j] = (distsq <= ri**2)

    m.append(np.copy(k_inh))
    k_inh = wi * k_inh / np.sum(k_inh)
    # m.append(np.copy(k_inh))

    return k_exc, k_inh

k_exc, k_inh = get_kernels(re, ri, wi, we, sigmaE)
exck = np.expand_dims(np.asarray(k_exc), axis = 0)
inhk = np.expand_dims(np.asarray(k_inh), axis = 0)

def get_laplacian(n, r1, r2, wi, we, sigmaE = 3):
    assert r1 < r2
    r1sq = r1**2
    r2sq = r2**2
    side_length = int(np.sqrt(n))
    f.extend([r1sq, r2sq, side_length])

    def find_distsq(i,j):
        xi, yi = i//side_length, i%side_length
        xj, yj = j//side_length, j%side_length
        return (xi-xj)**2+(yi-yj)**2
    # construct the W matrix
    We = np.zeros(shape = (n,n))
    Wi = np.zeros(shape = (n,n))
    for i in range(n):
        for j in range(n):
            # i row, j column
            distsq = find_distsq(i,j)
            f.append(distsq)
            We[i,j] = - we * np.exp(- distsq/2/sigmaE) * (distsq <= r1sq)
        We[i] = we * We[i] / -np.sum(We[i])
    m.append(np.copy(We))

    for i in range(n):
        for j in range(n):
            distsq = find_distsq(i,j)
            f.append(distsq)
            Wi[i,j] = wi * (distsq <= r2sq)
        Wi[i] = wi * Wi[i] /np.sum(Wi[i])
    m.append(np.copy(Wi))
    W = We + Wi
    np.fill_diagonal(W, we+wi)
    m.append(np.copy(W))
    return W

laplacian = get_laplacian(neuron_shape[0]*neuron_shape[1], r1 = re, r2 = ri, wi=wi, we=we)


def stimulate(stimulus, exc_act, inh_act):  # stimulus: (256, 20, 20)

    for t in range(50):
        exc_act_tm1 = np.copy(exc_act)

        #exc_act = exc_act + lr_act * (np.asarray(stimulus) - np.asarray(np.dot(exc_act, laplacian)))

        exc_input = convolve(exc_act, exck, mode="same")  # (256, 20, 20)
        inh_input = convolve(inh_act, inhk, mode="same")

        

        exc_act  = exc_act + lr_act * (- leaky * exc_act + stimulus + exc_input - inh_input)
        inh_act = inh_act + lr_act * (- leaky * inh_act + exc_input)

        # Soft threshold
        exc_act = np.maximum(exc_act - threshold, 0) - np.maximum(-exc_act - threshold, 0)
        inh_act = np.maximum(inh_act - threshold, 0) - np.maximum(-inh_act - threshold, 0)

        da = exc_act - exc_act_tm1


        relative_error = np.sqrt(np.square(da).sum()) / (eps + np.sqrt(np.square(exc_act_tm1).sum()))

        f.append(relative_error)

    # m.append(np.copy(exc_input))
    # m.append(np.copy(inh_input))
    # m.append(np.copy(exc_act))
    # m.append(np.copy(inh_act))
    # m.append(np.copy(da))

    if relative_error < eps:
        return exc_act
    else:
        print("error = " + str(relative_error))
        print("exc_act = "+ str(exc_act))
        print("Relative error end with {:.4f} and doesn't converge within the max fit steps".format(exc_act))
        return exc_act


Phi = 0.3 * np.random.rand(imbed_dim, neuron_shape[0]*neuron_shape[1])

word_embeddings = numpy.load('../../data/googleNgram/embed100.npy')
m.append(np.copy(word_embeddings))
word_embeddings = numpy.delete(word_embeddings, [55, 58, 84], axis=1)
word_embeddings = np.asarray(word_embeddings)
# m.append(np.copy(word_embeddings))

l2_loss = []
l1_loss = []
l0_loss = []
tbar = trange(initial_step, initial_step + gradient_steps, desc='Training', leave=True, miniters=100)

for i in tbar:
    word_batch = load_train_batch()  #this_X : 256 * 97

    stimulus = np.dot(word_batch, Phi).reshape((bs, neuron_shape[0], neuron_shape[1]))  # stimulus: (256, 20， 20)
    # Get neuron activity

    if i % 100 == 0:
        m.append(np.copy(stimulus))

    exc_act = np.zeros(shape=(bs, neuron_shape[0], neuron_shape[1]))
    inh_act = np.zeros(shape=(bs, neuron_shape[0], neuron_shape[1]))

    activation = stimulate(stimulus, exc_act, inh_act)
    activation = activation.reshape(bs, neuron_shape[0] * neuron_shape[1])

    if i % 100 == 0:
        m.append(np.copy(activation))

    # Neuron model evolve and reset
    dthreshold = np.mean((np.abs(activation) > 1e-4)) - l0_target
    threshold += .01 * dthreshold

    if i % 100 == 0:
        f.extend([dthreshold, threshold])

    ########### Update codebook

    fitted_value = np.dot(activation, np.transpose(Phi))
    if i % 100 == 0:
        m.append(np.copy(fitted_value))
    error = word_batch - fitted_value
    gradient = np.dot(np.transpose(error), activation)
    if i % 100 == 0:
        m.append(np.copy(gradient))
    Phi += lr_Phi * (gradient - np.vstack(np.mean(gradient, axis=1)))
    if i % 100 == 0:
        m.append(np.copy(Phi))
    Phi = Phi / np.maximum(np.sqrt(np.square(Phi).sum(axis=0)), 1e-8)
    # m.append(np.copy(Phi))

    l0l = np.mean(np.abs(activation) > 1e-4)
    l1l = np.abs(activation).mean()
    l2l = np.sqrt(np.square(error).sum())

    if i % 100 == 0:
        f.extend([l0l, l1l, l2l])

    # exc_act.fill(0)
    # inh_act.fill(0)

    ########### end

    l2_loss.append(float(l2l))
    l1_loss.append(float(l1l))
    l0_loss.append(float(l0l))

    if i % 100 == 0:
        #print(str(i) + ". Phi = " + str(Phi))
        tbar.set_description("loss=%.3f sparsity=%2.2f%% lmda=%.3f" % \
                                     (l2l, 100 * l0l, threshold))
        tbar.refresh()


print("finish training")

import pickle

def to_dict(f, value, isMore):
    f_dict = {}

    for x in f:
        x = round(x, 5)
        if isMore:
            if x >= value:
                if x not in f_dict.keys():
                    f_dict[x] = 1
                else:
                    f_dict[x] += 1
        else:
            if x < value:
                if x not in f_dict.keys():
                    f_dict[x] = 1
                else:
                    f_dict[x] += 1
    return f_dict

m1 = []
for arr in m:
    m1.extend(arr.flatten().tolist())

m1.extend(f)

with open("value_less", "wb") as file:
    pickle.dump(to_dict(m1, 35, False), file)

with open("value_more", "wb") as file:
    pickle.dump(to_dict(m1, 35, True), file)

print("value dumped")

