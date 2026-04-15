import pandas as pd
import os

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "cleaned_titanic.csv"))

#Analysis
print("Survival rate:", df["Survived"].mean())

print("Survival by Sex:")
print(df.groupby("Sex")["Survived"].mean())

print("Survival by Class:")
print(df.groupby("Pclass")["Survived"].mean())

print("Average Age by Survival:")
print(df.groupby("Survived")["Age"].mean())