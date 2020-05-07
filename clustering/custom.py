import pdb

import pandas
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11],

])

colors = ['b', 'r', 'g', 'y']


# plt.scatter(X[:,0], X[:,1])
# plt.show()

class KMeans:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}
        self.data = data
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classification = {}
            print(self.centroids)
            for z in range(self.k):
                self.classification[z] = []

            for featureset in self.data:
                distances = [
                    # use .items()
                    np.linalg.norm(featureset - centroid) for centroid in self.centroids.values()
                ]
                classification = distances.index(min(distances))
                self.classification[classification].append(featureset)

            previous_centroid = dict(self.centroids)

            for k, v in self.classification.items():
                self.centroids[k] = np.average(v, axis=0)
                # pass

            optimized = True
            # pdb.set_trace()

            for centroid_key, centroid_val in self.centroids.items():
                original_centroid = previous_centroid[centroid_key]
                current_centroid = centroid_val
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False
            if optimized:
                break

    def predict(self, data):
        distances = [
            np.linalg.norm(data - centroid) for centroid in self.centroids.values()
        ]
        classification = distances.index(min(distances))
        return classification


clf = KMeans()
clf.fit(X)

for k, v in clf.centroids.items():
    plt.scatter(v[0], v[1],
                marker='o', color='k', linewidths=5)
for classification in clf.classification:
    color = colors[classification]
    for feature in clf.classification[classification]:
        # pdb.set_trace()
        plt.scatter(feature[0], feature[1], marker='x', color=color, s=150, linewidths=5)

unkowns = np.array([
    [1, 3],
    [8, 9],
    [0, 3],
    [5, 4],
    [6, 4],
])

for un in unkowns:
    clasification = clf.predict(un)

    plt .scatter(un[0], un[1], marker='*', color=colors[clasification], s=150, linewidths=5)


plt.show()
