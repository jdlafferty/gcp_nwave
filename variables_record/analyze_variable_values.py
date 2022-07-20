import pickle

with open('value_less', 'rb') as f:
    small = pickle.load(f)
print(len(small))

with open('value_more', 'rb') as f:
    large = pickle.load(f)
print(len(large))

import matplotlib.pyplot as plt

print(min(small.keys()))
print(max(small.keys()))

print(min(large.keys()))
print(max(large.keys()))

plt.hist(list(small.keys()), bins=10000)
plt.title(f"Value smaller than 35\nmin={min(small.keys())}\nmax={max(small.keys())}")
plt.savefig("small_value.pdf")

plt.hist(list(large.keys()), bins=10000)
plt.title(f"Value larger than 35\nmin={min(large.keys())}\nmax={max(large.keys())}")
plt.savefig("large_value.pdf")
