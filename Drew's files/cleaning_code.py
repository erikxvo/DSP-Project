import pandas as pd

#Load data
df = pd.read_csv("Titanic-Dataset.csv")

#Quick Inspection
print(df.head())
print(df.info())
print(df.describe())

#CLEANING STEPS
#Removing column with too many missing values
df = df.drop(columns=["Cabin"])

# Fill missing values safely
df.loc[:, "Age"] = df["Age"].fillna(df["Age"].median())
df.loc[:, "Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical variables
df.loc[:, "Sex"] = df["Sex"].str.lower().map({"male": 0, "female": 1})

# One-hot encode Embarked
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Save cleaned dataset
df.to_csv("titanic_clean.csv", index=False)

print("Cleaning complete and file saved.")