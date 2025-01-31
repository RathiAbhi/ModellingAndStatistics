import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # Feature matrix
y = pd.Series(iris.target)  # Target labels

# Step 2: Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training size: {X_train.shape}, Testing size: {X_test.shape}")

# Step 3: Overfitting Example (Deep Decision Tree)
overfit_tree = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)  # No depth restriction
overfit_tree.fit(X_train, y_train)

# Step 4: Predict and Evaluate Overfitting Model
y_pred_overfit = overfit_tree.predict(X_test)
print("\nOverfitting Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_overfit):.3f}")
print(classification_report(y_test, y_pred_overfit, target_names=iris.target_names))

# Step 5: Correct Overfitting with Pruning (max_depth=3)
pruned_tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
pruned_tree.fit(X_train, y_train)

# Step 6: Predict and Evaluate Pruned Model
y_pred_pruned = pruned_tree.predict(X_test)
print("\nPruned Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_pruned):.3f}")
print(classification_report(y_test, y_pred_pruned, target_names=iris.target_names))

# Step 7: Visualizing the Overfitting Model
plt.figure(figsize=(12, 6))
plot_tree(overfit_tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Overfitting Decision Tree (Deep Tree)")
plt.show()

# Step 8: Visualizing the Pruned Model
plt.figure(figsize=(12, 6))
plot_tree(pruned_tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Pruned Decision Tree (max_depth=3)")
plt.show(block=True)