"""
Data Science Principles Project - Titanic Dataset
Ryan Hustis
Descriptive Visualization Plots
"""

import pandas as pd
import matplotlib as plt
import os


df = pd.read_csv(os.path.join(os.path.dirname(__file__), "cleaned_titanic.csv"))

out_dir = os.path.dirname(os.path.abspath(__file__))

# Plot 1 - Survival Rate by Gender

survival_rate_by_gender = df.groupby("Sex")["Survived"].mean()*100
labels = ["Male", "Female"]
colors = ["#750c35", "f40knn"]

fig, ax = plt.subplots(figsize=(7,5))
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
plot1_path = os.path.join(out_dir, "plot1_survival_by_gender.png")
plt.savefig(plot1_path, dpi=150)
plt.close()