from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_FILE = Path("data/team_14_train.csv")
FEATURES = [
    "battery_v",
    "temp_c",
    "rings_per_day",
    "wifi_rssi_dbm",
    "age_months",
]
TARGET = "days_left"


def main():
    df = pd.read_csv(DATA_FILE)

    print(f"rows={len(df)}, columns={list(df.columns)}")
    print("\nDescriptive statistics:")
    print(df.describe().to_string())

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print("\nUnique values:")
    print(df.nunique().to_string())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, column in zip(axes.flat, df.columns):
        ax.hist(df[column], bins=30, edgecolor="black")
        ax.set_title(column)
        ax.set_xlabel(column)
        ax.set_ylabel("count")
    axes.flat[-1].axis("off")
    fig.suptitle("Team 14: distributions of all columns")
    fig.tight_layout()
    fig.savefig("team14_column_histograms.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, column in zip(axes.flat, FEATURES):
        ax.scatter(df[column], df[TARGET], s=8, alpha=0.35)
        ax.set_title(f"{TARGET} vs {column}")
        ax.set_xlabel(column)
        ax.set_ylabel(TARGET)
    axes.flat[-1].axis("off")
    fig.suptitle("Team 14: days_left against each input")
    fig.tight_layout()
    fig.savefig("team14_days_left_vs_inputs.png", dpi=160)
    plt.close(fig)

    print("\nPearson correlations:")
    print(df.corr(numeric_only=True)[TARGET].sort_values(ascending=False).to_string())

    print("\nMean, median, minimum, and maximum days_left by feature fifth:")
    for column in FEATURES:
        groups = pd.qcut(df[column], q=5, duplicates="drop")
        summary = df.groupby(groups, observed=True)[TARGET].agg(
            ["count", "mean", "median", "min", "max"]
        )
        print(f"\n{column}:")
        print(summary.round(2).to_string())

    print("\nPotential extreme or invalid-looking values:")
    checks = {
        "battery_v == 0": df["battery_v"].eq(0),
        "temp_c < -40": df["temp_c"].lt(-40),
        "rings_per_day > 30": df["rings_per_day"].gt(30),
        "wifi_rssi_dbm == 0": df["wifi_rssi_dbm"].eq(0),
    }
    for label, mask in checks.items():
        print(f"{label}: {int(mask.sum())}")
        if mask.any():
            print(
                df.loc[mask, FEATURES + [TARGET]]
                .head(5)
                .to_string(index=False)
            )

    print("\nSaved team14_column_histograms.png")
    print("Saved team14_days_left_vs_inputs.png")


if __name__ == "__main__":
    main()
