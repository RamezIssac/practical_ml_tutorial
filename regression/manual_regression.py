from statistics import mean
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float)


def best_fit(xs, ys):
    nominator = (mean(xs) * mean(ys)) - mean(xs * ys)
    denominator = mean(xs) ** 2 - mean(xs ** 2)
    m = nominator / denominator
    b = mean(ys) - (m * mean(xs))
    return m, b


m, b = best_fit(xs, ys)

regression_line = [(m * x) + b for x in xs]
plt.scatter(xs,ys)
plt.plot(regression_line)
plt.show()