import numpy
#import numpy as np
import cupy as np
from cusignal.convolution.convolve import convolve
#from scipy.signal import convolve
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams['figure.figsize'] = (8, 6)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

ri = 5
re = 3
wi = 5
we = 30
leaky = wi + we
bs = 256
imbed_dim = 97
neuron_shape = (20, 20)
lr_act = 0.01
lr_Phi = 0.01
l0_target = 0.1
threshold = 0.01
initial_step = 0
gradient_steps = 10000
sigmaE = 3
max_act_fit = 50
eps = 5e-3

word_freq = numpy.load("../data/googleNgram/1gramSortedFreq.npy")
num_train_vocabs = word_freq.shape[0]
print("num_train_vocabs = " + str(num_train_vocabs))
SUBSAMPLE_SIZE = numpy.asarray(4096)

def load_train_batch():
    sampled_idx = np.random.choice(55529, bs, replace=False)
    word_batch = word_embeddings[sampled_idx,:]
    return word_batch

# def sample_word_idx():
#     subsampled_idx = numpy.random.randint(0, num_train_vocabs, SUBSAMPLE_SIZE)
#     prob = word_freq[subsampled_idx]
#     prob = prob / numpy.abs(prob).sum()
#     sampled_locs = numpy.random.choice(a=subsampled_idx, size=bs, replace=False, p=prob)
#     sampled_locs = np.asarray(sampled_locs)
#     return sampled_locs
#
# def load_train_batch():
#     sampled_idx = sample_word_idx()
#     word_batch = word_embeddings[sampled_idx, :]
#     return word_batch

def get_kernels(re, ri, wi=5, we=30, sigmaE = 3):
    k_exc = np.zeros([2*re+1, 2*re+1])
    k_inh = np.zeros([2*ri+1, 2*ri+1])
    for i in range(2*re+1):
        for j in range(2*re+1):
            # i row, j column
            distsq = (i-re)**2+(j-re)**2
            k_exc[i,j] = np.exp(- distsq/2/sigmaE) * (distsq <= re**2)
    k_exc = we * k_exc / np.sum(k_exc)
    for i in range(2*ri+1):
        for j in range(2*ri+1):
            # i row, j column
            distsq = (i-ri)**2+(j-ri)**2
            k_inh[i,j] = (distsq <= ri**2)
    k_inh = wi * k_inh / np.sum(k_inh)
    return k_exc, k_inh

k_exc, k_inh = get_kernels(re, ri, wi, we, sigmaE)
exck = np.expand_dims(np.asarray(k_exc), axis = 0)
inhk = np.expand_dims(np.asarray(k_inh), axis = 0)

def stimulate(stimulus, exc_act, inh_act):  # stimulus: (256, 20, 20)

    for t in range(50):
        exc_act_tm1 = np.copy(exc_act)

        exc_input = convolve(exc_act, exck, mode="same")  # (256, 20, 20)
        inh_input = convolve(inh_act, inhk, mode="same")

        exc_act = exc_act + lr_act * (- leaky * exc_act + stimulus + exc_input - inh_input)
        inh_act = inh_act + lr_act * (- leaky * inh_act + exc_input)

        # Soft threshold
        exc_act = np.maximum(exc_act - threshold, 0) - np.maximum(-exc_act - threshold, 0)
        inh_act = np.maximum(inh_act - threshold, 0) - np.maximum(-inh_act - threshold, 0)

        da = exc_act - exc_act_tm1
        relative_error = np.sqrt(np.square(da).sum()) / (eps + np.sqrt(np.square(exc_act_tm1).sum()))

    if relative_error < eps:
        return exc_act
    else:
        print("error = " + str(relative_error))
        print("exc_act = " + str(exc_act))
        print("Relative error end with {:.4f} and doesn't converge within the max fit steps".format(exc_act))
        return exc_act


Phi = 0.3 * np.random.rand(imbed_dim, neuron_shape[0]*neuron_shape[1])  #change 0.3

word_embeddings = numpy.load('../data/googleNgram/embed100.npy')
word_embeddings = numpy.delete(word_embeddings, [55, 58, 84], axis=1)
word_embeddings = np.asarray(word_embeddings)

l2_loss = []
l1_loss = []
l0_loss = []
tbar = trange(initial_step, initial_step + gradient_steps, desc='Training', leave=True, miniters=100)

exc_act = np.zeros(shape=(bs, neuron_shape[0], neuron_shape[1]))
inh_act = np.zeros(shape=(bs, neuron_shape[0], neuron_shape[1]))

for i in tbar:
    word_batch = load_train_batch()

    stimulus = np.dot(word_batch, Phi).reshape((bs, neuron_shape[0], neuron_shape[1]))  # stimulus: (256, 20， 20)
    # Get neuron activity

    activation = stimulate(stimulus, exc_act, inh_act)

    activation = activation.reshape(bs, neuron_shape[0] * neuron_shape[1])

    # Neuron model evolve and reset
    dthreshold = np.mean((np.abs(activation) > 1e-4)) - l0_target
    threshold += .01 * dthreshold

    ########### Update codebook


    fitted_value = np.dot(activation, np.transpose(Phi))
    error = word_batch - fitted_value
    gradient = np.dot(np.transpose(error), activation)
    Phi += lr_Phi * (gradient - np.vstack(np.mean(gradient, axis=1)))
    Phi = Phi / np.maximum(np.sqrt(np.square(Phi).sum(axis=0)), 1e-8)


    l0l = np.mean(np.abs(activation) > 1e-4)
    l1l = np.abs(activation).mean()
    l2l = np.sqrt(np.square(error).sum())

    exc_act.fill(0)
    inh_act.fill(0)

    ########### end

    l2_loss.append(float(l2l))
    l1_loss.append(float(l1l))
    l0_loss.append(float(l0l))

    if i % 100 == 0:
        tbar.set_description("loss=%.3f sparsity=%2.2f%% threshold=%.3f" % \
                                     (l2l, 100 * l0l, threshold))
        tbar.refresh()


batch = np.eye(imbed_dim)
batch = batch - numpy.mean(batch, axis=1)
batch = batch / numpy.std(batch, axis=1)

stimulus = np.dot(batch, Phi).reshape((batch.shape[0], neuron_shape[0], neuron_shape[1]))

exc_act1 = np.zeros(shape=(imbed_dim, neuron_shape[0], neuron_shape[1]))
inh_act1 = np.zeros(shape=(imbed_dim, neuron_shape[0], neuron_shape[1]))

RC = stimulate(stimulus, exc_act1, inh_act1)
RC = RC.reshape([imbed_dim, neuron_shape[0] * neuron_shape[1]])

def plot_colortable(colors, text_on=True):
    # ref: https://matplotlib.org/stable/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
    nunit = len(colors)
    side_length = int(numpy.sqrt(nunit))
    swatch_width = cell_width = cell_height = 32
    # set figs
    ncols = nrows = side_length
    width = cell_width * ncols
    height = cell_height * nrows
    dpi = 72
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    # set ax axis
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows), 0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for unit_idx, nunit in enumerate(range(nunit)):
        row = unit_idx // ncols
        col = unit_idx % ncols
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col  # + swatch_width

        if text_on:
            ax.text(text_pos_x + cell_width / 2, y + cell_height / 2, unit_idx, fontsize=6,
                    horizontalalignment='center', verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y), width=swatch_width, height=swatch_width, facecolor=colors[unit_idx],
                      edgecolor='0.7')
            )

    return fig

def rescale(vec, qt):
    qtmin = numpy.quantile(vec, qt, axis=1)[:, numpy.newaxis]
    qtmax = numpy.quantile(vec, 1 - qt, axis=1)[:, numpy.newaxis]
    return numpy.minimum(numpy.maximum((vec - qtmin) / (qtmax - qtmin), 0), 1)

def get_colors(Vt, alpha=0.5):
    _, n = Vt.shape
    colors = []
    for i in range(n):
        colors.append((*Vt[:, i], alpha))
    return colors

def plot_PCA(Phi, filename=''):
    Phi = np.asnumpy(Phi)
    U, S, Vt = numpy.linalg.svd(Phi.T, full_matrices=False)   # Phi: 97 * 400
    principal_score = U @ numpy.diag(S)[:, :3]
    principal_scoreT = rescale(principal_score.T, 0.05)
    colors = get_colors(principal_scoreT, alpha=0.8)
    fig = plot_colortable(colors, text_on=False)
    if len(filename) > 0:
        fig.savefig(filename, bbox_inches='tight')
    plt.close()

plot_PCA(RC, "RC.pdf")



