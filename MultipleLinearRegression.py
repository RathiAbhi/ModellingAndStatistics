import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Switch backend to MacOSX for macOS
import matplotlib
matplotlib.use('MacOSX')  # Alternatively, try 'Qt5Agg'

# Step 1: Create Synthetic Data
np.random.seed(42)  # For reproducibility
n_samples = 100

# Generate random data
X1 = np.random.rand(n_samples) * 10
X2 = np.random.rand(n_samples) * 5
X3 = np.random.rand(n_samples) * 20
y = 3.5 * X1 + 2.3 * X2 - 1.2 * X3 + 5 + np.random.randn(n_samples) * 2  # Linear relationship with noise

# Combine features into a DataFrame
data = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'y': y
})

# Step 2: Perform OLS Analysis
X = data[['X1', 'X2', 'X3']]
X_ols = sm.add_constant(X)  # Add intercept to the model
y = data['y']

ols_model = sm.OLS(y, X_ols).fit()  # Fit the OLS model
print(ols_model.summary())  # Print the OLS summary

# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression Model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = reg_model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Results:")
print("Coefficients:", reg_model.coef_)
print("Intercept:", reg_model.intercept_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Step 6: Visualize Results
plt.figure()  # Create a new figure
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.show()