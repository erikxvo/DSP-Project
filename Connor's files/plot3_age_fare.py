# Connor Smith
# Plot 3 = Average Fare by Age

# Interpretation = The line plot reveals that fare prices are quite volatile across ages, with several sharp spikes indicating that certain ages — particularly in the 30–50 range — include passengers who paid significantly higher fares. Overall there is no strong linear trend, meaning age alone does not consistently predict how much someone paid. However, the spikes likely correspond to wealthy First Class passengers at those ages, reinforcing that class and wealth were unevenly distributed across the age groups.

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Titanic-Dataset.csv")

avg_fare_by_age = df.groupby("Age")["Fare"].mean()

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(avg_fare_by_age.index, avg_fare_by_age.values, color="#378ADD", linewidth=2)

ax.set_title("Average Fare by Age", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Age", fontsize=12)
ax.set_ylabel("Average Fare (£)", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()