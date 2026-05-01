"""
Data Science Principles Project - Titanic Dataset
Ryan Hustis
Descriptive Visualization Plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("cleaned_titanic.csv")
os.makedirs("Ryan's files", exist_ok=True)

# Plot 1 - Survival Rate by Gender

survival_rate_by_gender = df.groupby("Sex")["Survived"].mean()*100
labels = ["Male", "Female"]
colors = ["#750c35", "#3a7ebf"]

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
plotryan1_path = "Ryan's files/plot1_survival_by_gender.png"
plt.savefig(plotryan1_path, dpi=150)
plt.close()

print("Plot 1 saved:", plotryan1_path)

# =============================================================================
# Plot 2 — Histogram: Fare Paid by Age
# =============================================================================
import numpy as np

fig, ax = plt.subplots(figsize=(7, 5))

df["Embarked"] = np.select(
    [df["Embarked_Q"] == 1, df["Embarked_S"] == 1],
    ["Queenstown", "Southampton"],
    default="Cherbourg"
)

survival_by_port = df.groupby("Embarked")["Survived"].mean() * 100
ports = ["Cherbourg", "Queenstown", "Southampton"]

ax.bar(ports, survival_by_port[ports], color="#3a7ebf", edgecolor="black", linewidth=0.4, width=0.5)

ax.set_title("Survival Rate by Embarkation Port", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Embarkation Port", fontsize=12)
ax.set_ylabel("Survival Rate (%)", fontsize=12)
ax.set_ylim(0, 80)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plot2_embarkedport_vs_survival = "Ryan's files/plot_survival_by_port.png"
plt.savefig(plot2_embarkedport_vs_survival, dpi=150)
plt.close()

print("Plot saved:", plot2_embarkedport_vs_survival)

# =============================================================================
# Plot 3: Survival Rate by Family Size
# =============================================================================

df["FamilySize"] = df["FamilySize"] + 1
survival_rate_by_family = df.groupby("FamilySize")["Survived"].mean() * 100

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(survival_rate_by_family.index, survival_rate_by_family.values,
       color="#8f7ebf", edgecolor="black", width=0.5)

for x, val in zip(survival_rate_by_family.index, survival_rate_by_family.values):
    ax.text(x, val + 1, f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

ax.set_title("Survival Rate by Family Size", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Family Size", fontsize=12)
ax.set_ylabel("Survival Rate (%)", fontsize=12)
ax.set_ylim(0, 100)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plotryan4_path = "Ryan's files/plot3_survival_by_familysize.png"
plt.savefig(plotryan4_path, dpi=150)
plt.close()

print("Plot 3 saved:", plotryan4_path)