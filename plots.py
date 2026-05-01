"""
Titanic Project — Descriptive and Diagnostic Plots
==================================================

Loads ``cleaned_titanic.csv`` and writes 12 PNG plots to the current
directory — every original plot from every teammate is preserved.
(The model's feature-importance bar chart lives in ``model.py``.)
Every figure is saved with ``plt.savefig`` — there are no ``plt.show()``
calls, so this script runs cleanly headless.

Run from the repo root:
    python3 plots.py

Contributors:
    Erik   — plots 1, 3, 4 (and shared figure styling)
    Connor — plots 2, 5, 11
    Drew   — plots 6, 9, 10
    Ryan   — plots 7, 8, 12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

DPI = 150
COLOR_SURVIVED = "#2ecc71"
COLOR_DIED = "#e74c3c"
COLOR_NEUTRAL = "#3a7ebf"

# Load the cleaned dataset once and share it across every plot function.
df = pd.read_csv("cleaned_titanic.csv")


# =============================================================================
# Plot 1 — Survival Rate by Passenger Class                       (author: Erik)
# =============================================================================

def plot_01_survival_by_class(df: pd.DataFrame) -> str:
    """Bar chart of survival rate (%) for each passenger class."""
    rates = df.groupby("Pclass")["Survived"].mean() * 100
    labels = ["1st Class\n(Upper)", "2nd Class\n(Middle)", "3rd Class\n(Lower)"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, rates.values, color=colors, edgecolor="black", width=0.5)

    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11)

    ax.set_title("Survival Rate by Passenger Class",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Passenger Class", fontsize=12)
    ax.set_ylabel("Survival Rate (%)", fontsize=12)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = "01_survival_by_class.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    return path


# =============================================================================
# Plot 2 — Survival Rate by Sex                                 (author: Connor)
# =============================================================================

def plot_02_survival_by_sex(df: pd.DataFrame) -> str:
    """Bar chart of survival rate (%) for each sex."""
    # In the cleaned data Sex is encoded 0=male, 1=female. Map back for labels.
    rates = (df.groupby("Sex")["Survived"].mean() * 100).rename(
        index={0: "Male", 1: "Female"}
    )
    colors = ["#378ADD", "#D4537E"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(rates.index, rates.values, color=colors, width=0.45,
                  edgecolor="none")

    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_title("Survival Rate by Sex", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Sex", fontsize=12)
    ax.set_ylabel("Survival Rate (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    plt.tight_layout()
    path = "02_survival_by_sex.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    return path


# =============================================================================
# Plot 3 — Age Distribution by Survival Status                    (author: Erik)
# =============================================================================

def plot_03_age_distribution_by_survival(df: pd.DataFrame) -> str:
    """Overlaid histograms of age, split by survival status."""
    survived = df.loc[df["Survived"] == 1, "Age"]
    not_survived = df.loc[df["Survived"] == 0, "Age"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(not_survived, bins=30, alpha=0.6, color=COLOR_DIED,
            edgecolor="black", linewidth=0.5, label="Did Not Survive")
    ax.hist(survived, bins=30, alpha=0.6, color=COLOR_SURVIVED,
            edgecolor="black", linewidth=0.5, label="Survived")

    ax.set_title("Age Distribution by Survival Status",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel("Number of Passengers", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = "03_age_distribution_by_survival.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    return path


# =============================================================================
# Plot 4 — Fare Paid by Survival Status                           (author: Erik)
# =============================================================================

def plot_04_fare_by_survival(df: pd.DataFrame) -> str:
    """Side-by-side boxplots of Fare for survivors vs non-survivors."""
    fare_died = df.loc[df["Survived"] == 0, "Fare"]
    fare_survived = df.loc[df["Survived"] == 1, "Fare"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [fare_died, fare_survived],
        tick_labels=["Did Not Survive", "Survived"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="gray",
                        markersize=4, alpha=0.5),
    )
    bp["boxes"][0].set_facecolor(COLOR_DIED)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(COLOR_SURVIVED)
    bp["boxes"][1].set_alpha(0.7)

    ax.set_title("Fare Paid by Survival Status",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Survival Status", fontsize=12)
    ax.set_ylabel("Fare (£)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = "04_fare_by_survival.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    return path


# =============================================================================
# Plot 5 — Fare Distribution by Passenger Class                 (author: Connor)
# =============================================================================

def plot_05_fare_distribution_by_class(df: pd.DataFrame) -> str:
    """Overlaid histograms of fare for each passenger class."""
    colors = ["#378ADD", "#D4537E", "#1D9E75"]
    labels = ["First Class", "Second Class", "Third Class"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for (pclass, group), color, label in zip(df.groupby("Pclass"), colors, labels):
        ax.hist(group["Fare"].dropna(), bins=30, alpha=0.7,
                color=color, edgecolor="white", linewidth=0.5, label=label)

    ax.set_title("Fare Distribution by Passenger Class",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Fare (£)", fontsize=12)
    ax.set_ylabel("Number of Passengers", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=11, framealpha=0.4)
    # Trim the x-axis to the 99th percentile so a few extreme fares don't
    # squash the rest of the distribution.
    ax.set_xlim(0, df["Fare"].quantile(0.99))

    plt.tight_layout()
    path = "05_fare_distribution_by_class.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    return path


# =============================================================================
# Plot 6 — Age vs Fare, Coloured by Survival                      (author: Drew)
# =============================================================================

def plot_06_age_vs_fare_by_survival(df: pd.DataFrame) -> str:
    """Scatter of Age vs Fare with survivor / non-survivor hue."""
    survived = df[df["Survived"] == 1]
    died = df[df["Survived"] == 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(died["Age"], died["Fare"], s=30, alpha=0.45,
               color=COLOR_DIED, edgecolor="black", linewidth=0.3,
               label="Did Not Survive")
    ax.scatter(survived["Age"], survived["Fare"], s=30, alpha=0.7,
               color=COLOR_SURVIVED, edgecolor="black", linewidth=0.3,
               label="Survived")

    ax.set_title("Age vs Fare, Coloured by Survival",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel("Fare (£)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = "06_age_vs_fare_by_survival.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    return path


# =============================================================================
# Plot 7 — Survival Rate by Family Size                          (author: Ryan)
# =============================================================================

def plot_07_survival_by_family_size(df: pd.DataFrame) -> str:
    """Bar chart of survival rate (%) across FamilySize values."""
    rates = df.groupby("FamilySize")["Survived"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rates.index, rates.values, color="#8f7ebf",
           edgecolor="black", width=0.6)

    for x, val in zip(rates.index, rates.values):
        ax.text(x, val + 1, f"{val:.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=10)

    ax.set_title("Survival Rate by Family Size",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Family Size (siblings/spouses + parents/children)",
                  fontsize=12)
    ax.set_ylabel("Survival Rate (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xticks(sorted(rates.index))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = "07_survival_by_family_size.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    return path


# =============================================================================
# Plot 8 — Survival Rate by Gender                                (author: Ryan)
# =============================================================================

def plot_08_survival_by_gender(df: pd.DataFrame) -> str:
    """Bar chart of survival rate by gender — Ryan's original styling."""
    rates = (df.groupby("Sex")["Survived"].mean() * 100).rename(
        index={0: "Male", 1: "Female"}
    )
    colors = ["#750c35", "#3a7ebf"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(rates.index, rates.values, color=colors,
                  edgecolor="black", width=0.5)

    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11)

    ax.set_title("Survival Rate by Gender", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Gender", fontsize=12)
    ax.set_ylabel("Survival Rate (%)", fontsize=12)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = "08_survival_by_gender.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    return path


# =============================================================================
# Plot 9 — Passenger Class Distribution                           (author: Drew)
# =============================================================================

def plot_09_passenger_class_distribution(df: pd.DataFrame) -> str:
    """Bar chart of raw passenger counts per class — Drew's original."""
    class_counts = df["Pclass"].value_counts().sort_index()
    labels = ["1st", "2nd", "3rd"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, class_counts.values)

    for bar, val in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 10,
                str(val), ha="center")

    ax.set_title("Passenger Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Passengers")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = "09_passenger_class_distribution.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    return path


# =============================================================================
# Plot 10 — Age Distribution (Overall)                            (author: Drew)
# =============================================================================

def plot_10_age_distribution_overall(df: pd.DataFrame) -> str:
    """Single histogram of overall passenger ages — Drew's original."""
    ages = df["Age"].dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ages, bins=25, edgecolor="black")

    ax.set_title("Age Distribution of Passengers")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = "10_age_distribution_overall.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    return path


# =============================================================================
# Plot 11 — Average Fare by Age                                 (author: Connor)
# =============================================================================

def plot_11_average_fare_by_age(df: pd.DataFrame) -> str:
    """Line plot of mean fare aggregated per age — Connor's original."""
    avg_fare_by_age = df.groupby("Age")["Fare"].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(avg_fare_by_age.index, avg_fare_by_age.values,
            color="#378ADD", linewidth=2)

    ax.set_title("Average Fare by Age", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Average Fare (£)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = "11_average_fare_by_age.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    return path


# =============================================================================
# Plot 12 — Survival Rate by Embarkation Port                    (author: Ryan)
# =============================================================================

def plot_12_survival_by_port(df: pd.DataFrame) -> str:
    """Bar chart of survival rate (%) for each embarkation port.

    Reconstructs the original Embarked column from the one-hot dummies
    (Embarked_Q, Embarked_S; Cherbourg is the implicit baseline).
    """
    embarked = np.select(
        [df["Embarked_Q"] == 1, df["Embarked_S"] == 1],
        ["Queenstown", "Southampton"],
        default="Cherbourg",
    )
    survival_by_port = pd.Series(df["Survived"].values).groupby(embarked).mean() * 100
    ports = ["Cherbourg", "Queenstown", "Southampton"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(ports, survival_by_port[ports], color="#3a7ebf",
           edgecolor="black", linewidth=0.4, width=0.5)

    ax.set_title("Survival Rate by Embarkation Port",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Embarkation Port", fontsize=12)
    ax.set_ylabel("Survival Rate (%)", fontsize=12)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = "12_survival_by_port.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    return path


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    plot_functions = [
        plot_01_survival_by_class,
        plot_02_survival_by_sex,
        plot_03_age_distribution_by_survival,
        plot_04_fare_by_survival,
        plot_05_fare_distribution_by_class,
        plot_06_age_vs_fare_by_survival,
        plot_07_survival_by_family_size,
        plot_08_survival_by_gender,
        plot_09_passenger_class_distribution,
        plot_10_age_distribution_overall,
        plot_11_average_fare_by_age,
        plot_12_survival_by_port,
    ]

    print(f"Saving {len(plot_functions)} plots to current directory ...")
    for fn in plot_functions:
        path = fn(df)
        print(f"  saved: {path}")
    print("All plots saved successfully.")


if __name__ == "__main__":
    main()
