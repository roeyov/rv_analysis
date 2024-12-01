import numpy as np
import matplotlib.pyplot as plt

noises = []
means = []
max_buckets = []
for i in range(1,10):

    deltas = []
    for j in range(10000):
        x = np.random.normal(0, i, 25)
        y = np.sort(x)
        deltas.append(y[-1] - y[0])
    # print(i, np.mean(deltas), np.median(deltas), np.std(deltas))
    n,bins,_ = plt.hist(deltas, bins=50, alpha=0.6, density=True)
    plt.xscale('log')
    noises.append(i)
    means.append(np.mean(deltas))
    max_buckets.append(np.max(n))

plt.show()

plt.scatter(noises, np.reciprocal(max_buckets))
plt.show()