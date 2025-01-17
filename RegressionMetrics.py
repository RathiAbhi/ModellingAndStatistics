import numpy as np
from scipy import stats

# Example data (independent and dependent variables)
x = np.array([50, 60, 50, 70, 65, 60])  # Independent variable
y = np.array([200, 300, 250, 400, 350, 280])  # Dependent variable

# Step 1: Add the intercept term to the design matrix
X = np.column_stack((np.ones(len(x)), x))  # Adds a column of ones for the intercept

# Step 2: Calculate the coefficients (beta)
# Formula: beta = (X.T @ X)^(-1) @ X.T @ y
beta = np.linalg.inv(X.T @ X) @ X.T @ y

# Step 3: Calculate predictions
y_pred = X @ beta

# Step 4: Calculate residuals
residuals = y - y_pred

# Step 5: Calculate variance and standard error
n = len(y)  # Number of data points
p = X.shape[1]  # Number of parameters (intercept + slope)
sigma_squared = np.sum(residuals**2) / (n - p)  # Variance of residuals
standard_error = np.sqrt(np.diagonal(sigma_squared * np.linalg.inv(X.T @ X)))  # Standard error for coefficients

# Step 6: Calculate t-statistics and p-values
t_stats = beta / standard_error
p_values = [2 * (1 - stats.t.cdf(np.abs(t), n - p)) for t in t_stats]

# Step 7: Calculate R-squared
ss_total = np.sum((y - np.mean(y))**2)  # Total sum of squares
ss_residual = np.sum(residuals**2)  # Residual sum of squares
r_squared = 1 - (ss_residual / ss_total)  # R-squared formula

# Print results
print("Coefficients (beta):", beta)
print("Standard Error:", standard_error)
print("t-Statistics:", t_stats)
print("P-values:", p_values)
print("R-squared:", r_squared)