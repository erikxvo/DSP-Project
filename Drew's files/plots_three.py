"""
Data Science Principles Project — Titanic Dataset
Part 2: Descriptive Visualizations (Drew's 3 Plots)
Author: Drew Cabral
"""

import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Load and prepare data
# =============================================================================

df = pd.read_csv("cleaned_titanic.csv")

# =============================================================================
# Plot 1 — Bar Plot: Passenger Class Distribution
# =============================================================================

class_counts = df["Pclass"].value_counts().sort_index()
labels = ["1st Class\n(Upper)", "2nd Class\n(Middle)", "3rd Class\n(Lower)"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, class_counts.values, edgecolor="black", width=0.5)

# Add value labels
for bar, val in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f"{val}", ha="center", va="bottom", fontweight="bold")

ax.set_title("Passenger Class Distribution", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Number of Passengers")
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("plot_class_distribution.png", dpi=150)
plt.close()

print("""
Interpretation:
Third-class passengers make up the largest portion of the dataset, while first-class
passengers represent the smallest group. This imbalance is important because it
means most passengers were in the lowest socioeconomic class, which likely
contributed to lower overall survival rates due to poorer cabin locations and
reduced access to lifeboats.
""")

# =============================================================================
# Plot 2 — Scatter Plot: Age vs Fare (colored by survival)
# =============================================================================

# Drop missing values for cleaner plotting
scatter_df = df[["Age", "Fare", "Survived"]].dropna()

survived = scatter_df[scatter_df["Survived"] == 1]
not_survived = scatter_df[scatter_df["Survived"] == 0]

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(not_survived["Age"], not_survived["Fare"],
           alpha=0.4, edgecolor="black", label="Did Not Survive")

ax.scatter(survived["Age"], survived["Fare"],
           alpha=0.7, edgecolor="black", label="Survived")

ax.set_title("Age vs Fare by Survival Status", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Age (years)")
ax.set_ylabel("Fare (£)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plot_scatter_age_fare.png", dpi=150)
plt.close()

print("""
Interpretation:
This scatter plot shows the relationship between age, fare, and survival. There is
no strong pattern between age and fare, but survival is more common among passengers
who paid higher fares. This suggests that wealth, rather than age, played a more
significant role in determining survival chances on the Titanic.
""")

# =============================================================================
# Plot 3 — Histogram: Age Distribution on the Titanic
# =============================================================================

ages = df["Age"].dropna()

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(ages, bins=30, edgecolor="black")

ax.set_title("Overall Age Distribution of Passengers", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Age (years)")
ax.set_ylabel("Number of Passengers")
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("plot_hist_age_distribution.png", dpi=150)
plt.close()

print("""
Interpretation:
The age distribution is right-skewed, with most passengers concentrated between
20 and 40 years old. There are relatively few very young children and elderly
passengers. This indicates that the Titanic primarily carried working-age adults,
which helps explain why this group experienced the highest number of casualties
during the disaster.
""")

# =============================================================================
# Overall Insight
# =============================================================================

print("""
Overall Insight:
Across all three plots, a consistent pattern emerges: socioeconomic status,
represented by passenger class and fare, strongly influenced survival outcomes.
Although most passengers were in lower classes, those who paid higher fares had
a clear advantage in survival, highlighting the role of inequality during the
disaster.
""")

print("All 3 plots saved successfully.")