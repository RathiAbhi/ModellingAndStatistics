import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
california_housing = fetch_california_housing()
X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
y = pd.Series(california_housing.target)

# Select features
X = X[['MedInc', 'AveRooms']]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 🚀 Sklearn Linear Regression ##
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)

# Metrics from sklearn
r2_sklearn = r2_score(y_test, y_pred_sklearn)
print("\n🔹 Sklearn Linear Regression 🔹")
print("R-squared:", r2_sklearn)
print("Intercept:", sklearn_model.intercept_)
print("Coefficients:", sklearn_model.coef_)

## 🚀 Statsmodels OLS Regression ##
X_train_sm = sm.add_constant(X_train)  # Add intercept manually
X_test_sm = sm.add_constant(X_test)

ols_model = sm.OLS(y_train, X_train_sm).fit()
y_pred_ols = ols_model.predict(X_test_sm)

# Metrics from statsmodels
print("\n🔹 Statsmodels OLS Regression 🔹")
print(ols_model.summary())  # Full statistical summary

# Extracting key statistics
print("\n🔹 Key Statsmodels OLS Metrics 🔹")
print("R-squared:", ols_model.rsquared)
print("Adjusted R-squared:", ols_model.rsquared_adj)
print("P-values:\n", ols_model.pvalues)
print("T-values:\n", ols_model.tvalues)
print("Confidence Intervals:\n", ols_model.conf_int())

# Plot residuals
plt.scatter(y_test, y_test - y_pred_ols)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show(block=True)