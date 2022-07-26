import pickle

with open('index', 'rb') as f:
    idx = pickle.load(f)
print(len(idx))

with open('value', 'rb') as f:
    value = pickle.load(f)
print(len(value))

import matplotlib.pyplot as plt

print(min(idx.keys()))
print(max(idx.keys()))

print(min(value.keys()))
print(max(value.keys()))

plt.hist(list(idx.keys()), bins=1000)
plt.title(f"Index range\nmin={min(idx.keys())}\nmax={max(idx.keys())}")
plt.xlim([min(idx.keys()) - 1, max(idx.keys()) + 1])
# plt.show()
plt.savefig("index.pdf")
plt.close()

plt.hist(list(value.keys()), bins=10000)
plt.title(f"Value range except for hyperparameters\nmin={min(value.keys())}\nmax={max(value.keys())}")
plt.xlim([min(value.keys()) - 1, max(value.keys()) + 1])
# plt.show()
plt.savefig("value.pdf")
plt.close()

