import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# Step 1: Generate synthetic dataset (non-linear relationship)
np.random.seed(42)
X = np.random.uniform(-3, 3, 100)  # Feature values
y = X**5 - 5*X**3 + 10*X + np.random.normal(0, 10, 100)  # Complex polynomial function with noise

# Convert X to 2D array
X = X.reshape(-1, 1)

# Step 2: Create polynomial features (high-degree polynomial to cause overfitting)
poly = PolynomialFeatures(degree=10)  # Very high-degree polynomial
X_poly = poly.fit_transform(X)

# Step 3: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Step 4: Train a normal Linear Regression model (expected to overfit)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions and R2 score
y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Linear Regression (Overfitting Model) R² - Train: {r2_train:.3f}, Test: {r2_test:.3f}")

# Step 5: Apply Ridge Regression (L2 Regularization)
ridge_model = Ridge(alpha=5)  # Add some penalty
ridge_model.fit(X_train, y_train)

y_train_ridge = ridge_model.predict(X_train)
y_test_ridge = ridge_model.predict(X_test)

r2_train_ridge = r2_score(y_train, y_train_ridge)
r2_test_ridge = r2_score(y_test, y_test_ridge)

print(f"Ridge Regression R² - Train: {r2_train_ridge:.3f}, Test: {r2_test_ridge:.3f}")

# Step 6: Apply Lasso Regression (L1 Regularization)
lasso_model = Lasso(alpha=0.1, max_iter=10000)  # Add L1 penalty
lasso_model.fit(X_train, y_train)

y_train_lasso = lasso_model.predict(X_train)
y_test_lasso = lasso_model.predict(X_test)

r2_train_lasso = r2_score(y_train, y_train_lasso)
r2_test_lasso = r2_score(y_test, y_test_lasso)

print(f"Lasso Regression R² - Train: {r2_train_lasso:.3f}, Test: {r2_test_lasso:.3f}")

# Step 7: Plot the models
plt.figure(figsize=(12, 5))
plt.scatter(X, y, label="True Data", color="black", alpha=0.5)

X_sorted = np.sort(X, axis=0)  # Sorting for visualization
X_sorted_poly = poly.transform(X_sorted)

plt.plot(X_sorted, linear_model.predict(X_sorted_poly), label="Linear (Overfit)", color="red", linestyle="dashed")
plt.plot(X_sorted, ridge_model.predict(X_sorted_poly), label="Ridge", color="blue")
plt.plot(X_sorted, lasso_model.predict(X_sorted_poly), label="Lasso", color="green")

plt.legend()
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.title("Regression Models: Overfitting vs Regularized Models")
plt.show(block=True)
