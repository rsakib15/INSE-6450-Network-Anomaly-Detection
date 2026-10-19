import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    DATA_PROCESSED,
    FIGURES,
    TABLES,
    save_table,
    save_fig,
    LABEL_COL,
    EXCLUDE_COLS,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def main():
    print("=" * 60)
    print("Exploratory Data Analysis")
    print("=" * 60)

    data_path = DATA_PROCESSED / "all_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run preprocessing.py first.")

    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Shape: {df.shape}")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        print("\nNo missing values found after cleaning.")
        miss_df = pd.DataFrame({"column": [], "missing_count": []})
    else:
        print("\nColumns with missing values:")
        print(missing)
        miss_df = missing.reset_index()
        miss_df.columns = ["column", "missing_count"]

        rows_with_missing = df[df.isnull().any(axis=1)]

        print("\nActual rows containing missing values:")
        print(rows_with_missing.to_string(index=False))

        output_path = DATA_PROCESSED / "rows_with_missing_values.csv"
        rows_with_missing.to_csv(output_path, index=False)

        print(f"\nTotal rows with missing values: {len(rows_with_missing):,}")
        print(f"Saved actual data to: {output_path}")
    save_table(miss_df, "missing_values.csv")

    n_dup_rows = df.duplicated().sum()
    print(f"\nExact duplicate rows: {n_dup_rows:,}")

    candidate_cols = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Fwd IAT Total",
        "Bwd Packet Length Std",
    ]

    num_cols = [c for c in candidate_cols if c in df.columns]

    if num_cols:
        stats = df[num_cols].describe().T
        stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        save_table(
            stats.reset_index().rename(columns={"index": "feature"}),
            "numeric_summary.csv",
        )
        print("\nNumeric summary (selected features) saved.")

    if "Flow Duration" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        vals = df["Flow Duration"].clip(upper=df["Flow Duration"].quantile(0.99))
        ax.hist(vals, bins=80, color="steelblue", edgecolor="white")
        ax.set_title("Flow Duration (clipped at 99th percentile)")
        ax.set_xlabel("Duration")
        ax.set_ylabel("Count")
        save_fig(fig, "flow_duration_hist.png")
        plt.close()

    if LABEL_COL in df.columns and "Bwd Packet Length Mean" in df.columns:
        df["is_attack"] = (df[LABEL_COL] != "BENIGN").astype(int)
        sample = df.sample(n=min(50_000, len(df)), random_state=42)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=sample, x="is_attack", y="Bwd Packet Length Mean", ax=ax)
        ax.set_xticklabels(["Benign", "Attack"])
        ax.set_title("Bwd Packet Length Mean – Benign vs Attack")
        save_fig(fig, "bwd_pkt_len_boxplot.png")
        plt.close()

    if len(num_cols) >= 4:
        corr = df[num_cols[:8]].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation – selected numeric features")
        save_fig(fig, "correlation_heatmap.png")
        plt.close()

    print("\nEDA finished. Artefacts written to outputs/")


if __name__ == "__main__":
    main()
