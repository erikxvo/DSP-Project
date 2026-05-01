"""
Titanic Project — Data Cleaning and Descriptive Analysis
========================================================

Reads the raw Titanic dataset, performs cleaning (drops, imputation,
encoding, feature engineering), prints a full set of descriptive
analyses, and writes the cleaned dataframe to ``cleaned_titanic.csv``.

Run from the repo root:
    python3 cleaning.py

Contributors:
    Erik   — pipeline structure, cleaning logic, descriptive outputs
    Connor — supplementary cleaning checks (superseded by Erik's pipeline)
"""

import pandas as pd
import numpy as np


# =============================================================================
# 1. Load Raw Dataset                                              (author: Erik)
# =============================================================================

df = pd.read_csv("Titanic-Dataset.csv")


# =============================================================================
# 2. Initial Inspection                                            (author: Erik)
# =============================================================================

print("=" * 60)
print("INITIAL DATA INSPECTION")
print("=" * 60)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Missing Values (Before Cleaning) ---")
missing_before = df.isnull().sum()
print(missing_before[missing_before > 0])


# =============================================================================
# 3. Cleaning                                                      (author: Erik)
# =============================================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Drop Cabin — ~77% of values are missing, not salvageable.
df.drop(columns=["Cabin"], inplace=True)
print("\n[1] Dropped 'Cabin' column (~77% missing — not usable).")

# Drop identifiers / free-text fields with no predictive value.
df.drop(columns=["Name", "Ticket", "PassengerId"], inplace=True)
print("[2] Dropped 'Name', 'Ticket', 'PassengerId' — not predictive.")

# Impute Age with median — median is robust to outliers in a skewed distribution.
age_median = df["Age"].median()
df["Age"] = df["Age"].fillna(age_median)
print(f"[3] Imputed missing 'Age' values with median ({age_median}).")

# Fill Embarked with mode — only 2 missing values.
embarked_mode = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(embarked_mode)
print(f"[4] Filled missing 'Embarked' values with mode ('{embarked_mode}').")

# Encode Sex as binary: male=0, female=1.
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
print("[5] Encoded 'Sex': male=0, female=1.")

# Encode Embarked as dummy variables (drop_first=True to avoid multicollinearity).
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
# Cast bool dummies to int so the saved CSV holds 0/1 instead of True/False.
for col in ["Embarked_Q", "Embarked_S"]:
    if col in df.columns:
        df[col] = df[col].astype(int)
print("[6] Encoded 'Embarked' as dummy variables (Embarked_Q, Embarked_S).")


# =============================================================================
# 4. Feature Engineering                                           (author: Erik)
# =============================================================================

# FamilySize captures total relatives aboard in a single feature.
df["FamilySize"] = df["SibSp"] + df["Parch"]
print("[7] Engineered 'FamilySize' = SibSp + Parch.")

print("\n--- Missing Values (After Cleaning) ---")
missing_after = df.isnull().sum()
remaining = missing_after[missing_after > 0]
print("No missing values remain." if remaining.empty else remaining)

print("\n--- Cleaned Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Cleaned Column List ---")
print(list(df.columns))


# =============================================================================
# 5. Descriptive Analyses                                          (author: Erik)
# =============================================================================

print("\n" + "=" * 60)
print("ANALYSIS OUTPUTS")
print("=" * 60)

total = len(df)

# [Analysis 1] Total passenger count.
print(f"\n[Analysis 1] Total passengers: {total}")

# [Analysis 2] Overall survival rate.
overall_survival_rate = df["Survived"].mean() * 100
print(f"[Analysis 2] Overall survival rate: {overall_survival_rate:.1f}%")

# [Analysis 3] Survivor count by sex.
survival_by_sex = df.groupby("Sex")["Survived"].sum()
survival_by_sex.index = survival_by_sex.index.map({0: "Male", 1: "Female"})
print(f"\n[Analysis 3] Survivors by sex:\n{survival_by_sex.to_string()}")

# [Analysis 4] Survival rate by sex.
survival_rate_by_sex = df.groupby("Sex")["Survived"].mean() * 100
survival_rate_by_sex.index = survival_rate_by_sex.index.map({0: "Male", 1: "Female"})
print(f"\n[Analysis 4] Survival rate by sex:\n{survival_rate_by_sex.round(1).to_string()}")

# [Analysis 5] Survival rate by passenger class.
survival_rate_by_pclass = df.groupby("Pclass")["Survived"].mean() * 100
survival_rate_by_pclass.index = ["1st Class", "2nd Class", "3rd Class"]
print(f"\n[Analysis 5] Survival rate by passenger class:\n"
      f"{survival_rate_by_pclass.round(1).to_string()}")

# [Analysis 6] Mean age by survival status.
mean_age = df.groupby("Survived")["Age"].mean()
mean_age.index = ["Did Not Survive", "Survived"]
print(f"\n[Analysis 6] Mean age by survival status:\n{mean_age.round(1).to_string()}")

# [Analysis 7] Mean fare by survival status.
mean_fare = df.groupby("Survived")["Fare"].mean()
mean_fare.index = ["Did Not Survive", "Survived"]
print(f"\n[Analysis 7] Mean fare by survival status:\n{mean_fare.round(2).to_string()}")

# [Analysis 8] Passenger count by embarkation port.
# Reconstruct port from dummies: neither Q nor S => Cherbourg (baseline).
def _port(row):
    if row.get("Embarked_Q", 0) == 1:
        return "Queenstown (Q)"
    if row.get("Embarked_S", 0) == 1:
        return "Southampton (S)"
    return "Cherbourg (C)"

df_port = df.apply(_port, axis=1)
print("\n[Analysis 8] Passenger count by embarkation port:")
for port, count in df_port.value_counts().items():
    print(f"  {port}: {count}")

# [Analysis 9] Survival rate by embarkation port.
print("\n[Analysis 9] Survival rate by embarkation port:")
for port in ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"]:
    mask = df_port == port
    if mask.any():
        rate = df.loc[mask, "Survived"].mean() * 100
        print(f"  {port}: {rate:.1f}%")

# [Analysis 10] Travel-group counts (alone vs with family).
alone = (df["FamilySize"] == 0).sum()
with_family = (df["FamilySize"] > 0).sum()
print("\n[Analysis 10] Travel group:")
print(f"  Travelling alone: {alone} ({alone / total * 100:.1f}%)")
print(f"  Travelling with family: {with_family} ({with_family / total * 100:.1f}%)")

# [Analysis 11] Survival rate: alone vs with family.
survival_alone = df.loc[df["FamilySize"] == 0, "Survived"].mean() * 100
survival_family = df.loc[df["FamilySize"] > 0, "Survived"].mean() * 100
print("\n[Analysis 11] Survival rate by travel group:")
print(f"  Travelling alone: {survival_alone:.1f}%")
print(f"  Travelling with family: {survival_family:.1f}%")

# [Analysis 12] Passenger count by sex.
total_male = (df["Sex"] == 0).sum()
total_female = (df["Sex"] == 1).sum()
print("\n[Analysis 12] Passenger count by sex:")
print(f"  Male:   {total_male} ({total_male / total * 100:.1f}%)")
print(f"  Female: {total_female} ({total_female / total * 100:.1f}%)")

# [Analysis 13] Median fare by passenger class.                    (author: Erik)
median_fare_by_class = df.groupby("Pclass")["Fare"].median().round(2)
median_fare_by_class.index = ["1st Class", "2nd Class", "3rd Class"]
print(f"\n[Analysis 13] Median fare by passenger class:\n"
      f"{median_fare_by_class.to_string()}")

# [Analysis 14] Correlation of every numeric feature with Survived. (author: Erik)
corr_with_survived = (
    df.corr(numeric_only=True)["Survived"].drop("Survived").round(3).sort_values()
)
print(f"\n[Analysis 14] Correlation with Survived (sorted):\n"
      f"{corr_with_survived.to_string()}")

# [Analysis 15] Summary statistics for Age and Fare.               (author: Erik)
print("\n[Analysis 15] Summary statistics for Age and Fare:")
print(df[["Age", "Fare"]].describe().round(2).to_string())


# =============================================================================
# 6. Save Cleaned Dataset                                          (author: Erik)
# =============================================================================

df.to_csv("cleaned_titanic.csv", index=False)
print("\n" + "=" * 60)
print("Cleaning and analysis complete.")
print("Cleaned dataset saved to: cleaned_titanic.csv")
print("=" * 60)
