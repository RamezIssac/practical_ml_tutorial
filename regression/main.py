import datetime

import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

# df = quandl.get('WIKI/GOOGL')
df = pd.read_pickle('df.pkl')

# df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
# df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
# df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)

forecast_out = math.ceil(0.01 * len(df))
import pdb

# pdb.set_trace()

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))

X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
y = np.array(df['label'][:-forecast_out])
# import pdb; pdb.set_trace()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
clf = pickle.load(open('classifier.pickle', 'rb'))
# clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)  # 0.9757257607536949

# clf = svm.SVR()
# clf.fit(X_train, y_train)
# accuracy = clf.score(X_test, y_test)
# print(accuracy)  # 0.8076620661853372

df['Forecast'] = np.nan

forecast_set = clf.predict(X_lately)
last_date = df.iloc[-1].name
one_day = 84600
next_date = last_date.timestamp() # + one_day
for i in forecast_set:
    df.loc[datetime.datetime.fromtimestamp(next_date)] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    next_date += one_day

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
