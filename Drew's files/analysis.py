import pandas as pd

<<<<<<< HEAD
df = pd.read_csv("titanic_clean.csv")
=======
df = pd.read_csv("cleaned_titanic.csv")
>>>>>>> df07ece526c2d52e2f790bf207905c6db5d0dc39

#Analysis
print("Survival rate:", df["Survived"].mean())

print("Survival by Sex:")
print(df.groupby("Sex")["Survived"].mean())

print("Survival by Class:")
print(df.groupby("Pclass")["Survived"].mean())

print("Average Age by Survival:")
print(df.groupby("Survived")["Age"].mean())