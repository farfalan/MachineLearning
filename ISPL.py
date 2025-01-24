from ISLP import load_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)

Boston = load_data('Boston')

df = pd.DataFrame(Boston)
X = df.iloc[:, 0:12]
y = df.iloc[:,12]

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)  

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

acuracy = model.score(X_test,y_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Residual Plot
plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=y_test - y_pred, lowess=True)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Actual vs. Predicted Plot
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs. Predicted')
plt.show()