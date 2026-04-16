import pandas as pd
import matplotlib.pyplot as plt

#Read Data
df = pd.read_csv("cleaned_titanic.csv")

#Count passenger classes
class_counts = df["Pclass"].values_counts().sort_index()

#Plot
class_counts.plot(kind="bar")

#Labels and title
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.xticks([0,1,2], ["1st", "2nd", "3rd"], rotation=0)
plt.grid(axis='y')
plt.show()

print("""
Interpretation (Plot 1 - Bar Plot: Passenger Class Distribution): The majority of passengers aboard the Titanic were in third class, with significantly fewer passengers in the second class and the smallest group in the first class. This distribution reflects the ship's role in transporting large numbers of lower-income travelers and immigrants, while a smaller portion of wealthier passengers occupied higher classes.
Because third class passengers made up such a large share of the total population, this imbalance is important to consider in later analyses, especially when comparing survival outcomes across social class. 
""")