"""
Titanic Project — Predictive Model (Logistic Regression)
========================================================

Loads ``cleaned_titanic.csv``, splits it into training and test sets,
fits a logistic regression to predict ``Survived``, and prints the full
suite of evaluation metrics requested by the project brief: accuracy,
precision, recall, F1, and a confusion matrix.

Run from the repo root (after ``cleaning.py`` has produced the CSV):
    python3 model.py

Contributor:
    Erik — model pipeline and evaluation
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# =============================================================================
# 1. Load Cleaned Data
# =============================================================================

df = pd.read_csv("cleaned_titanic.csv")


# =============================================================================
# 2. Define Features and Target
# =============================================================================

# Target: Survived (0 = died, 1 = survived).
# Features: every other numeric column produced by cleaning.py.
y = df["Survived"]
X = df.drop(columns=["Survived"])

print("Features used:", list(X.columns))
print(f"Total samples: {len(df)}")


# =============================================================================
# 3. Train / Test Split
# =============================================================================

# 80/20 split, stratified so the survival ratio matches in both partitions.
# random_state=42 fixes the shuffle so results are reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} rows")
print(f"Test set:     {len(X_test)} rows")


# =============================================================================
# 4. Train Logistic Regression
# =============================================================================

# max_iter raised because the default of 100 occasionally fails to converge
# on this feature scale. All other hyperparameters are scikit-learn defaults
# (L2 regularisation, C=1.0, solver='lbfgs').
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# =============================================================================
# 5. Evaluate
# =============================================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average="macro")
recall_macro = recall_score(y_test, y_pred, average="macro")
f1_macro = f1_score(y_test, y_pred, average="macro")

print("\n" + "=" * 60)
print("EVALUATION METRICS")
print("=" * 60)

print(f"\nAccuracy:  {accuracy * 100:.1f}%")
print(f"Precision (macro avg): {precision_macro:.3f}")
print(f"Recall    (macro avg): {recall_macro:.3f}")
print(f"F1 score  (macro avg): {f1_macro:.3f}")

print("\nConfusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                   Predicted Died   Predicted Survived")
print(f"Actual Died       {cm[0, 0]:>14}   {cm[0, 1]:>18}")
print(f"Actual Survived   {cm[1, 0]:>14}   {cm[1, 1]:>18}")

print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))


# =============================================================================
# 6. Coefficients (interpretation aid)
# =============================================================================

# Each coefficient is the log-odds change in survival per unit increase in
# the feature. Positive => raises survival probability; negative => lowers it.
coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
print("Feature coefficients (sorted):")
print(coefs.round(3).to_string())


# =============================================================================
# 7. Feature Importance Plot
# =============================================================================

# Save a horizontal bar chart of the coefficients alongside the metrics.
colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in coefs.values]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(coefs.index, coefs.values, color=colors, edgecolor="black")
ax.axvline(0, color="black", linewidth=0.8)

ax.set_title("Logistic Regression Coefficients (Feature Importance)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Coefficient (log-odds change)", fontsize=12)
ax.grid(axis="x", linestyle="--", alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("\nFeature importance plot saved: feature_importance.png")
