
# (Shell commands)
!apt-get -qq install graphviz > /dev/null
!pip install -q graphviz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import graphviz

# ---- 2. Load the dataset ----
# Make sure heart.csv is uploaded in the Colab "Files" panel
df = pd.read_csv("heart.csv")

print("First 5 rows:")
display(df.head())
print("\nShape of dataset:", df.shape)
print("\nMissing values in each column:")
print(df.isnull().sum())

# ---- 3. Split into features (X) and target (y) ----
target_column = "target"
X = df.drop(target_column, axis=1)
y = df[target_column]

print("\nFeature columns:", list(X.columns))
print("Target distribution:\n", y.value_counts())

# ---- 4. Train/Test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)

# =============================
# PART 1: Decision Tree Classifier
# =============================

# ---- 5. Train a basic Decision Tree ----
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)

# ---- 6. Evaluate on train and test ----
y_train_pred = dt_clf.predict(X_train)
y_test_pred = dt_clf.predict(X_test)

train_acc_dt = accuracy_score(y_train, y_train_pred)
test_acc_dt = accuracy_score(y_test, y_test_pred)

print("\n=== Decision Tree: Basic Model ===")
print("Train Accuracy:", train_acc_dt)
print("Test Accuracy :", test_acc_dt)
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))

# ---- 7. Visualize the Decision Tree using Graphviz ----
dot_data = export_graphviz(
    dt_clf,
    out_file=None,
    feature_names=X.columns,
    class_names=[str(cls) for cls in sorted(y.unique())],
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
# Renders and saves a PDF named "decision_tree_heart.pdf"
graph.render("decision_tree_heart", format="pdf", cleanup=True)
graph  # In Colab this will display the tree inline

# =============================
# PART 2: Analyze Overfitting & Control Tree Depth
# =============================

max_depth_values = range(1, 16)  # try tree depths from 1 to 15
train_accuracies = []
test_accuracies = []

for depth in max_depth_values:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_accuracies.append(dt.score(X_train, y_train))
    test_accuracies.append(dt.score(X_test, y_test))

# ---- 8. Plot train vs test accuracy vs depth ----
plt.figure(figsize=(8, 5))
plt.plot(max_depth_values, train_accuracies, marker="o", label="Train Accuracy")
plt.plot(max_depth_values, test_accuracies, marker="s", label="Test Accuracy")
plt.xlabel("Tree Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Overfitting Analysis (Depth vs Accuracy)")
plt.legend()
plt.grid(True)
plt.show()

# Choose a reasonable depth (where test accuracy is good and gap is small)
best_depth = max_depth_values[int(np.argmax(test_accuracies))]
print("\nSuggested max_depth based on this run:", best_depth)

# Train a new Decision Tree with controlled depth
dt_pruned = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_pruned.fit(X_train, y_train)
y_test_pred_pruned = dt_pruned.predict(X_test)

print("\n=== Decision Tree (Pruned) ===")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred_pruned))

# =============================
# PART 3: Random Forest & Comparison
# =============================

# ---- 9. Train a Random Forest ----
rf_clf = RandomForestClassifier(
    n_estimators=100,      # number of trees
    max_depth=None,       # trees can grow fully (will be controlled by ensemble)
    random_state=42
)
rf_clf.fit(X_train, y_train)

# ---- 10. Evaluate Random Forest ----
y_train_pred_rf = rf_clf.predict(X_train)
y_test_pred_rf = rf_clf.predict(X_test)

train_acc_rf = accuracy_score(y_train, y_train_pred_rf)
test_acc_rf = accuracy_score(y_test, y_test_pred_rf)

print("\n=== Random Forest ===")
print("Train Accuracy:", train_acc_rf)
print("Test Accuracy :", test_acc_rf)
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred_rf))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred_rf))

# ---- 11. Compare accuracies ----
print("\n=== Model Comparison (Test Accuracy) ===")
print(f"Decision Tree (basic):  {test_acc_dt:.4f}")
print(f"Decision Tree (pruned): {accuracy_score(y_test, y_test_pred_pruned):.4f}")
print(f"Random Forest:          {test_acc_rf:.4f}")

# =============================
# PART 4: Interpret Feature Importances
# =============================

importances = rf_clf.feature_importances_
feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n=== Feature Importances (Random Forest) ===")
display(feature_importance_df)

# Plot feature importances
plt.figure(figsize=(8, 6))
plt.barh(feature_importance_df["feature"], feature_importance_df["importance"])
plt.xlabel("Importance")
plt.title("Feature Importances (Random Forest)")
plt.gca().invert_yaxis()
plt.grid(axis="x")
plt.show()

# =============================
# PART 5: Cross-Validation Evaluation
# =============================

# ---- 12. Cross-validation for Decision Tree (pruned) ----
cv_scores_dt = cross_val_score(
    dt_pruned, X, y, cv=5, scoring="accuracy"
)
print("\n=== Cross-Validation: Decision Tree (Pruned) ===")
print("CV Scores:", cv_scores_dt)
print("Mean Accuracy:", cv_scores_dt.mean())
print("Std Dev:", cv_scores_dt.std())

# ---- 13. Cross-validation for Random Forest ----
cv_scores_rf = cross_val_score(
    rf_clf, X, y, cv=5, scoring="accuracy"
)
print("\n=== Cross-Validation: Random Forest ===")
print("CV Scores:", cv_scores_rf)
print("Mean Accuracy:", cv_scores_rf.mean())
print("Std Dev:", cv_scores_rf.std())

print("\nTask 5 pipeline completed ")
