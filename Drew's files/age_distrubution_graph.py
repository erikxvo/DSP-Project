import pandas as pd
import matplotlib.pyplot as plt

#Read Data
df = pd.read_csv("titanic_clean.csv")

#Plot 1: Age Distrubution
plt.hist(df["Age"])
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()