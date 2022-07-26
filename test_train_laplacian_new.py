import numpy
import numpy as np
#from cusignal.convolution.convolve import convolve
from scipy.signal import convolve
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
bs = 128
imbed_dim = 97
neuron_shape = (20, 20)
side_length = neuron_shape[0]
lr_act = 0.01
lr_Phi = 0.01
l0_target = 0.1
threshold = 0.01
initial_step = 0
gradient_steps = 25000
sigmaE = 3
max_act_fit = 50
eps = 5e-3

word_freq = numpy.load("../data/googleNgram/1gramSortedFreq.npy")
num_train_vocabs = word_freq.shape[0]
print("num_train_vocabs = " + str(num_train_vocabs))
SUBSAMPLE_SIZE = numpy.asarray(4096)

def load_train_batch():
    a = []
    for i in range(55529):
        a.append(i)
    a = np.asarray(a)
    sampled_idx = np.random.choice(a, bs, replace=False)
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

########################################

def get_num_nbs(r):
    count = 0
    for i in range((r+1)**2):
        xi = i // (r+1)
        yi = i % (r+1)
        distsq = xi**2 + yi**2
        if distsq <= r**2:
            count+=1

    num_nbs = (count-(r+1))*4 +1
    return num_nbs

num_E_nbs = get_num_nbs(re)  # 29

num_I_nbs = get_num_nbs(ri)  # 81


#print("num_E_nbs = " + str(num_E_nbs))
#print("num_I_nbs = " +str(num_I_nbs))

#####################################
# compute index set of each neurons
#####################################

def get_struct(r):
    rsq = r**2
    l = np.zeros(2*r+1)
    for i in range(2*r+1):
        count = 0
        for j in range(2*r+1):
            distsq = (i - r) ** 2 + (j - r) ** 2
            if distsq <= rsq:
                count += 1
        l[i] = count
    return l

def get_index_from_position(xi, yi, xj, yj, r):
    r = int(r)
    xi = int(xi)
    yi = int(yi)
    xj = int(xj)
    yj = int(yj)
    l = get_struct(r)
    core_index = (get_num_nbs(r)-1)/2
    if yi == yj:
        index = core_index + (xj - xi)
        index = int(index)
        return index
    else:
        diff = 0
        for i in range(int(abs(yj - yi)-1)):
            diff += l[r-i-1]
        if (yi > yj):
            index = core_index - (l[r]-1)/2 - diff - (l[r-(yi - yj)]+1)/2 + (xj - xi)
            index = int(index)
            return index
        elif (yi < yj):
            index = core_index + (l[r]-1)/2 + diff + (l[r-(yi - yj)]+1)/2 + (xj - xi)
            index = int(index)
            return index


def compute_indexset(r, num_nbs):
    set = np.zeros(shape=(neuron_shape[0]*neuron_shape[1], num_nbs))
    set = set.astype(int)
    set.fill(neuron_shape[0]*neuron_shape[1])
    #print(set)
    for i in range(neuron_shape[0]*neuron_shape[1]):
        xi = i // side_length
        yi = i % side_length
        for j in range(neuron_shape[0]*neuron_shape[1]):
            xj = j//side_length
            yj = j % side_length
            distsq = (xi - xj)**2 + (yi - yj)**2
            if distsq <= r**2:
                index = get_index_from_position(xi, yi, xj, yj, r)
                index = int(index)
                #print("index = " + str(index))
                set[i][int(index)] = j
                #print(set[i][index])
    return set

N_E = compute_indexset(re, num_E_nbs)
N_I = compute_indexset(ri, num_I_nbs)

#print("N_E = " +str(N_E))
# print("N_I = " +str(N_I))

#####################################
# compute weight kernels W_E and W_I
#####################################
W_E = np.zeros(num_E_nbs)
W_I = np.zeros(num_I_nbs)

count_E = 0
for i in range((2*re+1)**2):
    xi = i // (2*re+1)
    yi = i % (2*re+1)
    distsq = (xi - re)**2 + (yi - re)**2
    if distsq <= re**2:
        W_E[count_E] = np.exp(- distsq/2/sigmaE)
        count_E += 1

#print("count_E = " + str(count_E))
W_E = we * W_E / np.sum(W_E)
#print("W_E = " +str(W_E))

count_I = 0
for i in range((2*ri+1)**2):
    xi = i // (2 * ri + 1)
    yi = i % (2 * ri + 1)
    distsq = (xi - ri) ** 2 + (yi - ri) ** 2
    if distsq <= ri**2:
        W_I[count_I] = 1
        count_I += 1

#print("count_I = " + str(count_I))
W_I = wi * W_I / np.sum(W_I)
#print("W_I = " +str(W_I))

###########################
# Update algorithms
###########################

def activation_update(a):
    b = np.zeros(shape=(bs, neuron_shape[0] * neuron_shape[1]))
    for k in range(bs):
        r = np.zeros(neuron_shape[0]*neuron_shape[1])
        for i in range(neuron_shape[0]*neuron_shape[1]):
            r[i] = - leaky * a[k][i]
            for j in range(num_E_nbs):
                r[i] += W_E[j] * a[k][N_E[i][j]]
            for j in range(num_I_nbs):
                r[i] -= W_I[j] * a[k][N_I[i][j]]

        for i in range(neuron_shape[0]*neuron_shape[1]):
            a[k][i] = r[i]

    for k in range(bs):
        for n in range(neuron_shape[0]*neuron_shape[1]):
            b[k][n] = a[k][n]
    return b

########################################

def stimulate(stimulus, exc_act):  # stimulus: (256, 20, 20)

    for t in range(50):
        exc_act_tm1 = np.copy(exc_act)

        delta_a = activation_update(activation_dummy)
        # print("activation_dummy1 = " + str(activation_dummy))
        # print("delta_a = " +str(delta_a))

        exc_act = exc_act + lr_act * (np.asarray(stimulus) + np.asarray(delta_a))  # dimension problem

        exc_act = np.maximum(exc_act - threshold, 0) - np.maximum(-exc_act - threshold, 0)

        # print("exc_act1 = " + str(exc_act))
        # print("#######################################\n")
        for i in range(bs):
            for j in range(neuron_shape[0] * neuron_shape[1]):
                activation_dummy[i][j] = exc_act[i][j]

        da = exc_act - exc_act_tm1
        relative_error = np.sqrt(np.square(da).sum()) / (eps + np.sqrt(np.square(exc_act_tm1).sum()))

    if relative_error < eps:
        return exc_act
    else:
        print("error = " + str(relative_error))
        print("exc_act = "+ str(exc_act))
        print("Relative error end with {:.4f} and doesn't converge within the max fit steps".format(exc_act))
        return exc_act


Phi = 0.3 * np.random.rand(imbed_dim, neuron_shape[0]*neuron_shape[1])

word_embeddings = numpy.load('../data/googleNgram/embed100.npy')
word_embeddings = numpy.delete(word_embeddings, [55, 58, 84], axis=1)
word_embeddings = np.asarray(word_embeddings)

l2_loss = []
l1_loss = []
l0_loss = []
tbar = trange(initial_step, initial_step + gradient_steps, desc='Training', leave=True, miniters=100)

for i in tbar:
    word_batch = load_train_batch()  #this_X : 256 * 97

    stimulus = np.dot(word_batch, Phi)  # stimulus: (256, 20， 20)
    # Get neuron activity

    exc_act = np.zeros(shape=(bs, neuron_shape[0]*neuron_shape[1]))
    inh_act = np.zeros(shape=(bs, neuron_shape[0]*neuron_shape[1]))

    activation_dummy = np.zeros(shape = (bs, neuron_shape[0]*neuron_shape[1]+1))

    activation = stimulate(stimulus, exc_act)
    #activation = activation.reshape(bs, neuron_shape[0] * neuron_shape[1])

    # Neuron model evolve and reset
    dthreshold = np.mean((np.abs(activation) > 1e-4)) - l0_target
    threshold += .01 * dthreshold

    ########### Update codebook

    #print("activ_shape = " +str(np.shape(activation)))

    fitted_value = np.dot(activation, np.transpose(Phi))
    error = word_batch - fitted_value
    gradient = np.dot(np.transpose(error), activation)
    Phi += lr_Phi * (gradient - np.vstack(np.mean(gradient, axis=1)))
    Phi = Phi / np.maximum(np.sqrt(np.square(Phi).sum(axis=0)), 1e-8)

    l0l = np.mean(np.abs(activation) > 1e-4)
    l1l = np.abs(activation).mean()
    l2l = np.sqrt(np.square(error).sum())

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

np.save("codebook_laplacian.npy", Phi)

################
#plot phi
################

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
    U, S, Vt = numpy.linalg.svd(Phi.T, full_matrices=False)   # Phi: 97 * 400
    principal_score = U @ numpy.diag(S)[:, :3]
    principal_scoreT = rescale(principal_score.T, 0.05)
    colors = get_colors(principal_scoreT, alpha=0.8)
    fig = plot_colortable(colors, text_on=False)
    if len(filename) > 0:
        fig.savefig(filename, bbox_inches='tight')
    plt.close()

plot_PCA(Phi, "Phi.pdf")

