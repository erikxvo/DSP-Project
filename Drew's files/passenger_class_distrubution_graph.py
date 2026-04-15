import pandas as pd
import matplotlib.pyplot as plt
import os

#Read Data
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "cleaned_titanic.csv"))

#Plot 2: Passager Class Distrubution
df["Pclass"].value_counts().plot(kind="bar")
plt.title("Passenger Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
