"""
Data Science Principles Project — Titanic Dataset
Part 1: Data Cleaning and Analysis
Author: Erik Vo
"""

import pandas as pd
import numpy as np

# =============================================================================
# 1. Load Dataset
# =============================================================================

df = pd.read_csv("Titanic-Dataset.csv")

# =============================================================================
# 2. Initial Inspection
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
# 3. Data Cleaning
# =============================================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Drop Cabin — ~77% of values are missing, not salvageable
df.drop(columns=["Cabin"], inplace=True)
print("\n[1] Dropped 'Cabin' column (~77% missing — not usable).")

# Drop Name, Ticket, PassengerId — not predictive features
df.drop(columns=["Name", "Ticket", "PassengerId"], inplace=True)
print("[2] Dropped 'Name', 'Ticket', 'PassengerId' — not predictive.")

# Impute Age with median — median is robust to outliers
age_median = df["Age"].median()
df["Age"] = df["Age"].fillna(age_median)
print(f"[3] Imputed missing 'Age' values with median ({age_median}).")

# Fill Embarked with mode — only 2 missing values
embarked_mode = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(embarked_mode)
print(f"[4] Filled missing 'Embarked' values with mode ('{embarked_mode}').")

# Encode Sex as binary: male=0, female=1
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
print("[5] Encoded 'Sex': male=0, female=1.")

# Encode Embarked as dummy variables (drop_first to avoid multicollinearity)
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
print("[6] Encoded 'Embarked' as dummy variables (Embarked_Q, Embarked_S).")

print("\n--- Missing Values (After Cleaning) ---")
missing_after = df.isnull().sum()
remaining = missing_after[missing_after > 0]
if remaining.empty:
    print("No missing values remain.")
else:
    print(remaining)

print("\n--- Cleaned Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Cleaned Column List ---")
print(list(df.columns))

# =============================================================================
# 4. Analysis Outputs (≥10 distinct metrics)
# =============================================================================

print("\n" + "=" * 60)
print("ANALYSIS OUTPUTS")
print("=" * 60)

# 1. Total passenger count
total = len(df)
print(f"\n[Analysis 1] Total passengers: {total}")

# 2. Overall survival rate
overall_survival_rate = df["Survived"].mean() * 100
print(f"[Analysis 2] Overall survival rate: {overall_survival_rate:.1f}%")

# 3. Survival count by sex
# Sex is now encoded: 0=male, 1=female
survival_by_sex = df.groupby("Sex")["Survived"].sum()
survival_by_sex.index = survival_by_sex.index.map({0: "Male", 1: "Female"})
print(f"\n[Analysis 3] Survivors by sex:\n{survival_by_sex.to_string()}")

# 4. Survival rate by sex
survival_rate_by_sex = df.groupby("Sex")["Survived"].mean() * 100
survival_rate_by_sex.index = survival_rate_by_sex.index.map({0: "Male", 1: "Female"})
print(f"\n[Analysis 4] Survival rate by sex:\n{survival_rate_by_sex.round(1).to_string()}")

# 5. Survival rate by passenger class
survival_rate_by_pclass = df.groupby("Pclass")["Survived"].mean() * 100
survival_rate_by_pclass.index = ["1st Class", "2nd Class", "3rd Class"]
print(f"\n[Analysis 5] Survival rate by passenger class:\n{survival_rate_by_pclass.round(1).to_string()}")

# 6. Mean age of survivors vs non-survivors
mean_age = df.groupby("Survived")["Age"].mean()
mean_age.index = ["Did Not Survive", "Survived"]
print(f"\n[Analysis 6] Mean age by survival status:\n{mean_age.round(1).to_string()}")

# 7. Mean fare of survivors vs non-survivors
mean_fare = df.groupby("Survived")["Fare"].mean()
mean_fare.index = ["Did Not Survive", "Survived"]
print(f"\n[Analysis 7] Mean fare by survival status:\n{mean_fare.round(2).to_string()}")

# 8. Passenger count by embarkation port
# Reconstruct port from dummies: if neither Q nor S → C; if Q → Q; if S → S
embarked_s_col = "Embarked_S" if "Embarked_S" in df.columns else None
embarked_q_col = "Embarked_Q" if "Embarked_Q" in df.columns else None

port_counts = {"Cherbourg (C)": 0, "Queenstown (Q)": 0, "Southampton (S)": 0}
for _, row in df.iterrows():
    if embarked_q_col and row[embarked_q_col] == 1:
        port_counts["Queenstown (Q)"] += 1
    elif embarked_s_col and row[embarked_s_col] == 1:
        port_counts["Southampton (S)"] += 1
    else:
        port_counts["Cherbourg (C)"] += 1

print("\n[Analysis 8] Passenger count by embarkation port:")
for port, count in port_counts.items():
    print(f"  {port}: {count}")

# 9. Survival rate by embarkation port
port_survival = {}
for _, row in df.iterrows():
    if embarked_q_col and row[embarked_q_col] == 1:
        port = "Queenstown (Q)"
    elif embarked_s_col and row[embarked_s_col] == 1:
        port = "Southampton (S)"
    else:
        port = "Cherbourg (C)"
    port_survival.setdefault(port, []).append(row["Survived"])

print("\n[Analysis 9] Survival rate by embarkation port:")
for port, values in port_survival.items():
    rate = np.mean(values) * 100
    print(f"  {port}: {rate:.1f}%")

# 10. Passengers travelling alone vs with family
df["FamilySize"] = df["SibSp"] + df["Parch"]
alone = (df["FamilySize"] == 0).sum()
with_family = (df["FamilySize"] > 0).sum()
print(f"\n[Analysis 10] Travel group:")
print(f"  Travelling alone: {alone} ({alone/total*100:.1f}%)")
print(f"  Travelling with family: {with_family} ({with_family/total*100:.1f}%)")

# 11. Survival rate: alone vs with family
survival_alone = df[df["FamilySize"] == 0]["Survived"].mean() * 100
survival_family = df[df["FamilySize"] > 0]["Survived"].mean() * 100
print(f"\n[Analysis 11] Survival rate by travel group:")
print(f"  Travelling alone: {survival_alone:.1f}%")
print(f"  Travelling with family: {survival_family:.1f}%")

# 12. Count of male vs female passengers
total_male = (df["Sex"] == 0).sum()
total_female = (df["Sex"] == 1).sum()
print(f"\n[Analysis 12] Passenger count by sex:")
print(f"  Male: {total_male} ({total_male/total*100:.1f}%)")
print(f"  Female: {total_female} ({total_female/total*100:.1f}%)")

print("\n" + "=" * 60)
print("Cleaning and analysis complete.")
print("=" * 60)

# =============================================================================
# 5. Save Cleaned Dataset
# =============================================================================

df.to_csv("cleaned_titanic.csv", index=False)
print("\nCleaned dataset saved to: cleaned_titanic.csv")
