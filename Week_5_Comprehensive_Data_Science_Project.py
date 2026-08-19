# ============================================================
# WEEK 5: COMPREHENSIVE DATA SCIENCE PROJECT
# Reporting and Strategic Recommendations
#
# Dataset:
# Breast Cancer Wisconsin Diagnostic Dataset
#
# Algorithm:
# Logistic Regression
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Dataset name: Breast Cancer Wisconsin Diagnostic Dataset")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nTarget classes:")
print(data.target_names)

print("\nFeature names:")
print(data.feature_names)


# ------------------------------------------------------------
# 3. CHECK DATA
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA CHECK")
print("=" * 60)

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)

print("\nClass distribution:")
unique, counts = np.unique(y, return_counts=True)

for label, count in zip(unique, counts):
    print(
        f"{label} ({data.target_names[label]}): {count}"
    )


# ------------------------------------------------------------
# 4. SPLIT DATA
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ------------------------------------------------------------
# 5. CREATE MACHINE LEARNING PIPELINE
# ------------------------------------------------------------

model = Pipeline([
    
    # Feature scaling
    (
        "scaler",
        StandardScaler()
    ),

    # Logistic Regression classifier
    (
        "classifier",
        LogisticRegression(
            max_iter=5000,
            random_state=42
        )
    )
])


# ------------------------------------------------------------
# 6. TRAIN MODEL
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

model.fit(
    X_train,
    y_train
)

print("Logistic Regression model trained successfully.")


# ------------------------------------------------------------
# 7. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)

# Probability predictions
y_probability = model.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------------------
# 8. CALCULATE EVALUATION METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ------------------------------------------------------------
# 9. DISPLAY MODEL PERFORMANCE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ------------------------------------------------------------
# 10. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)


# ------------------------------------------------------------
# 11. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ------------------------------------------------------------
# 12. CONFUSION MATRIX VISUALIZATION
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

plt.imshow(
    cm
)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.xticks(
    [0, 1],
    ["Malignant", "Benign"]
)

plt.yticks(
    [0, 1],
    ["Malignant", "Benign"]
)


# Add values inside confusion matrix
for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14
        )


plt.colorbar()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 13. ROC CURVE
# ------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)


# ------------------------------------------------------------
# 14. ROC CURVE VISUALIZATION
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

# Random classifier
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Logistic Regression"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 15. MODEL PERFORMANCE VISUALIZATION
# ------------------------------------------------------------

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

metric_values = [
    accuracy,
    precision,
    recall,
    f1,
    roc_auc
]


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    metric_names,
    metric_values
)

plt.ylim(
    0,
    1.05
)

plt.ylabel(
    "Score"
)

plt.title(
    "Logistic Regression Performance Metrics"
)

plt.xticks(
    rotation=20
)


# Display values above bars
for i, value in enumerate(metric_values):

    plt.text(
        i,
        value + 0.015,
        f"{value:.3f}",
        ha="center"
    )


plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16. FIVE-FOLD CROSS-VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("Cross-validation scores:")

for i, score in enumerate(
    cv_scores,
    start=1
):

    print(
        f"Fold {i}: {score:.4f}"
    )


cv_mean = cv_scores.mean()

cv_std = cv_scores.std()

print(
    f"\nMean CV Accuracy: {cv_mean:.4f}"
)

print(
    f"CV Standard Deviation: {cv_std:.4f}"
)


# ------------------------------------------------------------
# 17. CROSS-VALIDATION VISUALIZATION
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

fold_numbers = range(
    1,
    len(cv_scores) + 1
)

plt.plot(
    fold_numbers,
    cv_scores,
    marker="o",
    label="Fold Accuracy"
)

plt.axhline(
    cv_mean,
    linestyle="--",
    label=f"Mean Accuracy = {cv_mean:.3f}"
)

plt.xlabel(
    "Cross-Validation Fold"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "5-Fold Cross-Validation Accuracy"
)

plt.xticks(
    fold_numbers
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 18. STRATEGIC ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STRATEGIC RECOMMENDATIONS")
print("=" * 60)

print("""
1. External Validation
   Test the model on an independent dataset before considering
   real-world implementation.

2. Error Analysis
   Examine false positives and false negatives separately.
   Different errors may have different practical consequences.

3. Model Comparison
   Compare Logistic Regression with Decision Trees,
   Random Forest, Support Vector Machines, and other
   appropriate classification algorithms.

4. Hyperparameter Tuning
   Use GridSearchCV or RandomizedSearchCV to identify
   suitable model parameters.

5. Cross-Validation
   Continue using stratified cross-validation to obtain
   more reliable estimates of model performance.

6. Model Monitoring
   If deployed, continuously monitor data quality,
   prediction performance, and data/model drift.

7. Probability Calibration
   Evaluate whether predicted probabilities accurately
   represent estimated risk.

8. Stakeholder Communication
   Present model results together with limitations,
   uncertainty, and potential sources of error.
""")


# ------------------------------------------------------------
# 19. FINAL PROJECT SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print(
    "Project: Week 5 Comprehensive Data Science Project"
)

print(
    "Dataset: Breast Cancer Wisconsin Diagnostic Dataset"
)

print(
    "Algorithm: Logistic Regression"
)

print(
    f"Test Accuracy: {accuracy:.2%}"
)

print(
    f"Precision: {precision:.2%}"
)

print(
    f"Recall: {recall:.2%}"
)

print(
    f"F1 Score: {f1:.2%}"
)

print(
    f"ROC-AUC: {roc_auc:.2%}"
)

print(
    f"Mean 5-Fold CV Accuracy: {cv_mean:.2%}"
)

print(
    "\nComprehensive data science analysis completed successfully!"
)