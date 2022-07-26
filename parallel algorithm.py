##########################
# Initialize activations
##########################
import numpy as np

neuron_shape = (40, 40)
side_length = neuron_shape[0]
bs = 2
re = 3
ri = 5
sigmaE = 3
we = 30
wi = 5
leaky = we + wi

a = 0.007 * np.random.rand(bs, neuron_shape[0]*neuron_shape[1])
a = np.append(a, [[0],[0]], axis = 1)

print("a = " +str(a))

#####################################
# Compute num_E_nbs and num_I_nbs
#####################################

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


print("num_E_nbs = " + str(num_E_nbs))
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
#print("N_I = " +str(N_I))

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
print("W_E = " +str(W_E))

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


b = activation_update(a)

print("b = " +str(b))



