
df = pd.read_csv("Titanic-Dataset.csv")

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