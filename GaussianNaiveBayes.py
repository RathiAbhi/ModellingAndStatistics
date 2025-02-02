import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# 📌 Load Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train Gaussian Naïve Bayes Model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 📌 Predict
y_pred = gnb.predict(X_test)

# 📌 Evaluate Model
print("Gaussian Naïve Bayes Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))