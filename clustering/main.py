import pdb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import preprocessing
from sklearn.model_selection import cross_validate

style.use('ggplot')

df = pd.read_excel('titanic.xls')
df.drop(['name', 'body'], 1, inplace=True)
df = df.apply(pd.to_numeric, errors='ignore')
df.fillna(0, inplace=True)

# pd.get_dummies(df, columns=['cabin', 'home.dest', 'sex'], drop_first=True)
df['sex'] = pd.factorize(df['sex'])[0]
# df['cabin'] = pd.factorize(df['cabin'])[0]
# df['boat'] = pd.factorize(df['boat'])[0]
# df['home.dest'] = pd.factorize(df['home.dest'])[0]
df.drop(['home.dest', 'boat', 'ticket', 'sibsp', 'embarked', 'parch',], 1, inplace=True)

df['survived'] = pd.factorize(df['survived'])[0]
for c in df.columns:
    if df[c].dtype not in [np.int, np.float]:
        df[c] = pd.factorize(df[c])[0]

x = np.array(df.drop(['survived'], 1).astype(float))
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(x, y)

# test
correct = 0
for i in range(len(x)):
    predict = np.array(x[i])
    predict = predict.reshape(-1, len(predict))
    predictions = clf.predict(predict)
    if predictions[0] == y[i]:
        correct += 1

print(correct/len(x))
