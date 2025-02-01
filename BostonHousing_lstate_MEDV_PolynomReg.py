from ISLP import load_data
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)
#test push
# Load the Boston dataset
Boston = load_data('Boston')

# Create a DataFrame
df = pd.DataFrame(Boston)
X = df.iloc[:, 11:12]
y = df.iloc[:, 12]

# Define the parameter grid for polynomial degrees
param_grid = {
    'poly_features__degree': np.arange(1, 10)  # Test polynomial degrees from 1 to 9
}

# Plotting setup
plt.figure(figsize=(14, 8))

mse_values = []

# Loop through each degree in the parameter grid
for degree in param_grid['poly_features__degree']:
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    # Fit linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    coefficients = lin_reg.coef_
    intercept = lin_reg.intercept_
    polynomial_equation = f"{intercept:.2f}"
    for i, coef in enumerate(coefficients[1:], start=1):
        polynomial_equation += f" + ({coef:.2f} * X^{i})"
    print(f"Polynomial equation for degree {degree}: {polynomial_equation}")
    
    # Predict
    y_pred = lin_reg.predict(X_poly)
    
    # Calculate performance metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Degree {degree} - Mean Squared Error: {mse:.2f}, R² Score: {r2:.2f}")
    
    # Store MSE value
    mse_values.append(mse)
    
    # Plot
    plt.scatter(X, y_pred, label=f'Degree {degree}')

# Plot original data
plt.scatter(X, y, color='black', label='Original data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression with Different Degrees')
plt.legend()
plt.show()

# Plot MSE values
plt.figure(figsize=(10, 6))
plt.plot(param_grid['poly_features__degree'], mse_values, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error for Different Polynomial Degrees')
plt.show()

