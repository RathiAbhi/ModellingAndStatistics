import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

# Load the breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# Split the train and test dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=23
)

# Logistic Regression
clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:, 1]  # Probability predictions

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print("Logistic Regression model accuracy (in %):", acc * 100)

# Plot 1: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show(block=True)

# Plot 2: Confusion Matrix
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix")
plt.show(block=True)

# Scatter Plot of Predictions
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.7, label='True Labels')
plt.scatter(range(len(y_test)), y_pred_prob, color='orange', alpha=0.7, label='Predicted Probabilities')
plt.xlabel('Sample Index')
plt.ylabel('Values (0 or 1)')
plt.title('Scatter Plot of Predictions vs True Labels')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
plt.show(block=True)