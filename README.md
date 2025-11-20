# ⭐ **Task 5 — Decision Trees & Random Forests (Heart Disease Classification)**

This repository contains **Task 5** of my AIML Internship project.
The objective of this task is to build, visualize, and evaluate **tree-based machine learning models**—specifically **Decision Trees** and **Random Forests**—using the **Heart Disease dataset**.

This task explores model interpretability, overfitting analysis, feature importance, and model comparison.

---

## 📁 **Repository Structure**

```
├── heart.csv                      # Original dataset (uploaded)
├── heart_processed.csv            # Cleaned & preprocessed dataset
├── heart_decision_trees.py        # Complete training script (single-run)
├── README.md                      # Documentation (this file)
└── output/
    ├── decision_tree_heart.png        # Visualized decision tree (Graphviz)
    ├── depth_vs_accuracy.png          # Overfitting analysis (tree depth vs accuracy)
    ├── feature_importances.png        # Random Forest feature importance chart
    └── empty                          # Auto-created placeholder by Colab
```

---

## 🎯 **Objective**

Implement and understand the following:

* **Decision Tree Classifier**
* **Tree visualization using Graphviz**
* **Overfitting & depth control**
* **Random Forest Classifier**
* **Feature importance analysis**
* **Model evaluation using cross-validation**

---

## 🧹 **Data Preprocessing Steps**

Steps performed on the raw `heart.csv` dataset:

1. Loaded the dataset and checked for missing values.
2. Dropped rows containing null values (dataset had very few).
3. Separated features and target (`target` column).
4. Split the data into **train** and **test** sets (80/20).
5. Saved the cleaned dataset as `heart_processed.csv`.

---

## 🤖 **Model Training (heart_decision_trees.py)**

The script performs the entire machine learning pipeline:

### **1. Decision Tree Classifier**

* Initial model trained without constraints
* Tree visualized and exported as `decision_tree_heart.png`
* Performance evaluated on both train & test sets

### **2. Overfitting Analysis**

* Tested tree depths from **1 to 15**
* Plotted **Train Accuracy vs Test Accuracy**
  → Saved as `depth_vs_accuracy.png`
* Identified optimal depth for a pruned tree
* Retrained Decision Tree with best depth

### **3. Random Forest Classifier**

* Trained a 100-tree ensemble model
* Compared accuracy vs Decision Tree
* Extracted feature importances
  → Saved as `feature_importances.png`

### **4. Cross-Validation**

Evaluated both models using **5-fold cross-validation** for more reliable performance scores.

---

## 📊 **Generated Visualizations**

All stored inside the `output/` folder:

### ✔ **Decision Tree Visualization**

`decision_tree_heart.png`
Shows actual splits, thresholds, Gini values, and class distribution.

### ✔ **Depth vs Accuracy Plot**

`depth_vs_accuracy.png`
Shows overfitting behavior as tree depth increases.

### ✔ **Feature Importance Plot**

`feature_importances.png`
Ranks the most influential clinical features in predicting heart disease.

---

## 🧪 **Model Evaluation Metrics**

Evaluation includes:

* **Accuracy (Train & Test)**
* **Classification Report**
* **Confusion Matrix**
* **Cross-validation mean accuracy**

The Random Forest consistently outperforms the Decision Tree in stability and generalization.

---

## 🚀 **How to Run the Project**

### **Option 1 — Google Colab (Recommended)**

Upload:

* `heart.csv`
* `heart_decision_trees.py`

Run:

```python
!python heart_decision_trees.py
```

All outputs will be generated inside the **output/** folder.

---

### **Option 2 — Local Machine**

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib graphviz
```

Make sure **Graphviz** is installed on your system.

Run:

```bash
python heart_decision_trees.py
```

---

## 📝 **Dataset**

**Heart Disease Dataset**
A widely used medical dataset for binary classification (presence/absence of heart disease).
Features include clinical measurements such as cholesterol levels, resting BP, fasting blood sugar, etc.

---

## ✨ **Author**

**Thrishool M S**

AIML Internship — *Task 5: Decision Trees & Random Forests*

---
