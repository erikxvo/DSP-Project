"""
Data Science Principles Project — Titanic Dataset
Part 2: Descriptive Visualizations (Erik's 3 Plots)
Author: Erik Vo
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================================================================
# Load and prepare data
# =============================================================================

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "cleaned_titanic.csv"))

# Output directory (same folder as this script)
out_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Plot 1 — Bar Plot: Survival Rate by Passenger Class
# =============================================================================

survival_rate_by_class = df.groupby("Pclass")["Survived"].mean() * 100
labels = ["1st Class\n(Upper)", "2nd Class\n(Middle)", "3rd Class\n(Lower)"]
colors = ["#2ecc71", "#f39c12", "#e74c3c"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, survival_rate_by_class.values, color=colors, edgecolor="black", width=0.5)

# Add value labels on bars
for bar, val in zip(bars, survival_rate_by_class.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

ax.set_title("Survival Rate by Passenger Class", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Passenger Class", fontsize=12)
ax.set_ylabel("Survival Rate (%)", fontsize=12)
ax.set_ylim(0, 80)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plot1_path = os.path.join(out_dir, "plot1_survival_by_class.png")
plt.savefig(plot1_path, dpi=150)
plt.close()

print("Plot 1 saved:", plot1_path)
print("""
Interpretation (Plot 1 — Bar Plot: Survival Rate by Passenger Class):
Passenger class was one of the strongest predictors of survival. First-class
passengers survived at roughly 63%, compared to about 47% for second class and
only 24% for third class. This stark gradient reflects both socioeconomic privilege
(better cabin locations closer to lifeboats) and the social norms of the era, where
wealthier passengers were prioritized during the evacuation.
""")

# =============================================================================
# Plot 2 — Histogram: Age Distribution by Survival Status
# =============================================================================

survived = df[df["Survived"] == 1]["Age"]
not_survived = df[df["Survived"] == 0]["Age"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(not_survived, bins=30, alpha=0.6, color="#e74c3c", edgecolor="black",
        label="Did Not Survive", linewidth=0.5)
ax.hist(survived, bins=30, alpha=0.6, color="#2ecc71", edgecolor="black",
        label="Survived", linewidth=0.5)

ax.set_title("Age Distribution by Survival Status", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Age (years)", fontsize=12)
ax.set_ylabel("Number of Passengers", fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plot2_path = os.path.join(out_dir, "plot2_age_distribution.png")
plt.savefig(plot2_path, dpi=150)
plt.close()

print("Plot 2 saved:", plot2_path)
print("""
Interpretation (Plot 2 — Histogram: Age Distribution by Survival Status):
The overlapping histograms reveal that young children (under ~10) had a noticeably
higher survival proportion relative to their total count, consistent with a
"women and children first" evacuation policy. The largest cluster of non-survivors
falls in the 20–35 age range — predominantly young adult males travelling in
third class. Older passengers (60+) show lower survival counts, likely due to
limited mobility during the evacuation.
""")

# =============================================================================
# Plot 3 — Boxplot: Fare Paid by Survival Status
# =============================================================================

fare_not_survived = df[df["Survived"] == 0]["Fare"]
fare_survived = df[df["Survived"] == 1]["Fare"]

fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot(
    [fare_not_survived, fare_survived],
    tick_labels=["Did Not Survive", "Survived"],
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.5)
)
bp["boxes"][0].set_facecolor("#e74c3c")
bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor("#2ecc71")
bp["boxes"][1].set_alpha(0.7)

ax.set_title("Fare Paid by Survival Status", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Survival Status", fontsize=12)
ax.set_ylabel("Fare (£)", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plot3_path = os.path.join(out_dir, "plot3_fare_by_survival.png")
plt.savefig(plot3_path, dpi=150)
plt.close()

print("Plot 3 saved:", plot3_path)
print("""
Interpretation (Plot 3 — Boxplot: Fare Paid by Survival Status):
Survivors paid substantially higher fares than those who did not survive: the median
fare for survivors is roughly double that of non-survivors, and the interquartile
range extends much higher. Since fare is closely tied to passenger class and cabin
proximity to the lifeboats, this plot reinforces that economic status played a
critical role in survival outcomes. The many high-fare outliers among survivors
correspond to wealthy first-class ticket holders.
""")

print("All 3 plots saved successfully.")
