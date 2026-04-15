import pandas as pd

df = pd.read_csv("titanic_clean.csv")

#Analysis
print("Survival rate:", df["Survived"].mean())

print("Survival by Sex:")
print(df.groupby("Sex")["Survived"].mean())

print("Survival by Class:")
print(df.groupby("Pclass")["Survived"].mean())

print("Average Age by Survival:")
print(df.groupby("Survived")["Age"].mean())