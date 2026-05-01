# Connor Smith
# Plot 2 = Fare Distribution by Passenger Class

# Interpretation = The histogram shows that the vast majority of passengers paid low fares, with Third Class tickets clustered tightly under £20 and Second Class fares only slightly higher. First Class fares, by contrast, were spread across a much wider range — with some passengers paying over £200 — reflecting the significant wealth gap on board. This matters because fare and class were closely tied to survival: wealthier passengers in higher classes had cabins closer to the deck and better access to lifeboats.

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Titanic-Dataset.csv")

fig, ax = plt.subplots(figsize=(9, 5))

colors = ["#378ADD", "#D4537E", "#1D9E75"]
labels = ["First Class", "Second Class", "Third Class"]

for (pclass, group), color, label in zip(df.groupby("Pclass"), colors, labels):
    ax.hist(group["Fare"].dropna(), bins=30, alpha=0.7,
            label=label, color=color, edgecolor="white", linewidth=0.5)

ax.set_title("Fare Distribution by Passenger Class", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Fare (£)", fontsize=12)
ax.set_ylabel("Number of Passengers", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(fontsize=11, framealpha=0.4)
ax.set_xlim(0, df["Fare"].quantile(0.99))

plt.tight_layout()
plt.show()