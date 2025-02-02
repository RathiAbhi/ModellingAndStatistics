import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import make_classification

# Generate synthetic dataset
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# LightGBM model
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)

# Evaluate accuracy
xgb_acc = accuracy_score(y_test, xgb_preds)
lgb_acc = accuracy_score(y_test, lgb_preds)

print(f"XGBoost Accuracy: {xgb_acc:.4f}")
print(f"LightGBM Accuracy: {lgb_acc:.4f}")