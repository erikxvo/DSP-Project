"""
Data Science Principles Project - Titanic Dataset
Ryan Hustis
Descriptive Visualization Plots
"""

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("cleaned_titanic.csv")

# Plot 1 - Survival Rate by Gender

survival_rate_by_gender = df.groupby("Sex")["Survived"].mean()*100
labels = ["Male", "Female"]
colors = ["#750c35", "#3a7ebf"]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, survival_rate_by_gender.values, color=colors, edgecolor="black", width = 0.5)

for bar, val in zip(bars, survival_rate_by_gender.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

ax.set_title("Survival Rate by Gender", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Gender", fontsize=12)
ax.set_ylabel("Survival Rate (%)", fontsize=12)
ax.set_ylim(0, 80)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plotryan1_path = "Ryan's files/plot1_survival_by_gender.png"
plt.savefig(plotryan1_path, dpi=150)
plt.close()

print("Plot 1 saved:", plotryan1_path)