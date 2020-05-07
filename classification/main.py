import pandas
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

df = pandas.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -999, inplace=True)
df.drop(df.columns[0], 1, inplace=True)
# df.drop(df.columns[1], 1, inplace=True)
# df.drop(df.columns[2], 1, inplace=True)

X = np.array(df.drop(df.columns[-1], 1))
y = np.array(df[df.columns[-1]])

x_train, x_test, y_train, y_test = train_test_split(X, y)
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)