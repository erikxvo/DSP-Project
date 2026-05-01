# Connor Smith
# Plot 1 = Survival Rate by Sex

# Interpretation = Females had a survival rate of 74.2% compared to just 18.9% for males, revealing a dramatic difference in survival outcomes by sex. This is largely explained by the "women and children first" evacuation protocol followed by the Titanic's crew, which prioritised women's access to the limited lifeboats. This makes sex one of the strongest predictors of survival in the dataset.

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Titanic-Dataset.csv")

survival_rate = df.groupby("Sex")["Survived"].mean() * 100
print(survival_rate)

fig, ax = plt.subplots(figsize=(6, 5))
colors = ["#D4537E", "#378ADD"]
bars = ax.bar(survival_rate.index, survival_rate.values, color=colors, width=0.45, edgecolor="none")

for bar, val in zip(bars, survival_rate.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_title("Survival Rate by Sex", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Sex", fontsize=12)
ax.set_ylabel("Survival Rate (%)", fontsize=12)
ax.set_ylim(0, 100)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

plt.tight_layout()
plt.show()