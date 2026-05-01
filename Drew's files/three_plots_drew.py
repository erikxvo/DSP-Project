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
labels = ["1st", "2nd", "3rd"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, class_counts.values)

# Add simple labels
for bar, val in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 10,
            str(val), ha="center")

ax.set_title("Passenger Class Distribution")
ax.set_xlabel("Class")
ax.set_ylabel("Number of Passengers")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plot_class_distribution.png")
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

fig, ax = plt.subplots(figsize=(7, 4))

ax.scatter(not_survived["Age"], not_survived["Fare"],
           alpha=0.4, label="Did Not Survive")

ax.scatter(survived["Age"], survived["Fare"],
           alpha=0.7, label="Survived")

ax.set_title("Age vs Fare by Survival")
ax.set_xlabel("Age")
ax.set_ylabel("Fare")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plot_scatter_age_fare.png")
plt.close()

print("""
Interpretation:
There is no strong relationship between age and fare. However, passengers who paid
higher fares appear more likely to survive, suggesting that wealth played a larger
role in survival than age.
""")

# =============================================================================
# Plot 3 — Histogram: Age Distribution on the Titanic
# =============================================================================

ages = df["Age"].dropna()

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ages, bins=25, edgecolor="black")

ax.set_title("Age Distribution of Passengers")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plot_hist_age_distribution.png")
plt.close()

print("""
Interpretation:
The age distribution is right-skewed, with most passengers centered between
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
Looking at these plots together, passenger class and fare seem to have had a significant impact
on survival. Even though most passengers were in third class, people who paid higher
fares generally had a better chance of surviving, showing how social class likely
influenced outcomes during the disaster.
""")

print("All 3 plots saved successfully.")