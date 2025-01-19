from ISLP import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)

Boston = load_data('Boston')

X= Boston['lstat']
X = sm.add_constant(X)
y = Boston['medv']
model = sm.OLS(y, X)
results = model.fit()


ax = Boston.plot.scatter('lstat', 'medv')

xlim = ax.get_xlim ()
ylim = [results.params[0] + results.params[1] * x for x in xlim]

plt.plot(xlim, ylim, color='red')
plt.show()