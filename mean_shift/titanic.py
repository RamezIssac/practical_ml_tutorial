import pdb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans, MeanShift
from sklearn.datasets import make_blobs
from sklearn import preprocessing
from sklearn.model_selection import cross_validate

style.use('ggplot')

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['name', 'body'], 1, inplace=True)
df = df.apply(pd.to_numeric, errors='ignore')
df.fillna(0, inplace=True)

# pd.get_dummies(df, columns=['cabin', 'home.dest', 'sex'], drop_first=True)
df['sex'] = pd.factorize(df['sex'])[0]
# df['cabin'] = pd.factorize(df['cabin'])[0]
# df['boat'] = pd.factorize(df['boat'])[0]
# df['home.dest'] = pd.factorize(df['home.dest'])[0]
df.drop(['home.dest', 'boat', 'ticket', 'sibsp', 'embarked', 'parch', ], 1, inplace=True)

df['survived'] = pd.factorize(df['survived'])[0]
for c in df.columns:
    if df[c].dtype not in [np.int, np.float]:
        df[c] = pd.factorize(df[c])[0]

x = np.array(df.drop(['survived'], 1).astype(float))
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(x, y)

labels = clf.labels_
cluster_centers = clf.cluster_centers_
original_df['cluster_group'] = np.nan

for i in range(len(x)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters = len(np.unique(labels))

survival_rates = {}
# pdb.set_trace()
original_df.groupby('cluster_group').size()

for i in range(n_clusters):
    cluster_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = cluster_df[(cluster_df['survived'] == 1)]
    survival_rate = len(survival_cluster) / len(cluster_df)
    survival_rates[i] = survival_rate

print(survival_rates)

