"""
Data Science Principles Project — Titanic Dataset
Part 3: Predictive Model (Logistic Regression)
Author: Erik Vo
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# =============================================================================
# 1. Load Cleaned Data
# =============================================================================

df = pd.read_csv("cleaned_titanic.csv")

# Convert boolean dummy columns to int (get_dummies wrote them as True/False)
df["Embarked_Q"] = df["Embarked_Q"].astype(int)
df["Embarked_S"] = df["Embarked_S"].astype(int)

# =============================================================================
# 2. Define Features and Target
# =============================================================================

# Target: Survived (0 = died, 1 = survived)
# Features: every other column
y = df["Survived"]
X = df.drop(columns=["Survived"])

print("Features used:", list(X.columns))
print(f"Total samples: {len(df)}")

# =============================================================================
# 3. Train / Test Split
# =============================================================================

# 80/20 split, stratified so survival ratio is the same in both sets.
# random_state fixes the shuffle so results are reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} rows")
print(f"Test set:     {len(X_test)} rows")

# =============================================================================
# 4. Train Logistic Regression
# =============================================================================

# max_iter raised because default (100) sometimes doesn't converge on this data.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =============================================================================
# 5. Evaluate
# =============================================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {accuracy * 100:.1f}%")

print("\nConfusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predicted Died   Predicted Survived")
print(f"Actual Died      {cm[0,0]:>14}   {cm[0,1]:>18}")
print(f"Actual Survived  {cm[1,0]:>14}   {cm[1,1]:>18}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

# =============================================================================
# 6. Feature Importance (Logistic Regression Coefficients)
# =============================================================================

# Each coefficient shows how that feature shifts the log-odds of survival.
# Positive = increases survival probability; negative = decreases.
coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
print("\nFeature coefficients (sorted):")
print(coefs.round(3).to_string())

# =============================================================================
# 7. Plot Feature Importance
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in coefs.values]
ax.barh(coefs.index, coefs.values, color=colors, edgecolor="black")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Logistic Regression Coefficients (Feature Importance)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Coefficient (log-odds change)", fontsize=12)
ax.grid(axis="x", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("Erik's files/plot4_feature_importance.png", dpi=150)
plt.close()
print("\nFeature importance plot saved: Erik's files/plot4_feature_importance.png")

print("""
Interpretation:
The strongest positive predictor of survival is Sex (female=1) — being female
sharply raised the odds of survival. Pclass has the strongest negative
coefficient: higher class number (3rd > 1st) lowers survival probability,
matching the descriptive plots. Age and FamilySize contribute smaller
negative effects (older passengers and very large families fared worse).
Fare and Embarked dummies have small effects once class is already in the model.
""")
