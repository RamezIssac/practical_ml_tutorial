import pdb

import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs

style.use('ggplot')

centers = np.array([
    [1, 2, 3],
    [5, 5, 5],
    [3, 10, 10],
])

X, _ = make_blobs(n_samples=1000, centers=centers, cluster_std=1)
ms = MeanShift()
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
n_clusters = len(np.unique(labels))
print('Numbers of clusters', n_clusters)

colors = 10 * ['b', 'r', 'g', 'y']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print(len(X))

# this scatter the data points
for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

# this scatter the centers
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='x', zorder=10, linewidth=5,
           s= 150, color='k')
plt.show()
