import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    DATA_PROCESSED,
    LABEL_COL,
    list_raw_csvs,
    save_table,
    save_fig,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("Removing empty rows...")
    df = df.dropna(how="all").copy()
    df.columns = [str(column).strip() for column in df.columns]

    print("Removing duplicate columns...")
    df = df.loc[:, ~df.columns.duplicated()]

    print("Cleaning network traffic rate columns...")
    for column in ["Flow Bytes/s", "Flow Packets/s"]:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan)
            df[column] = values.fillna(-1)

    print("Normalizing the label column...")
    label_column = next(
        (column for column in df.columns if column.lower() == "label"),
        None,
    )

    if label_column is not None:
        if label_column != LABEL_COL:
            df = df.rename(columns={label_column: LABEL_COL})

        df[LABEL_COL] = (
            df[LABEL_COL]
            .astype(str)
            .str.replace("–", "-", regex=False)
            .str.replace("—", "-", regex=False)
            .str.replace("\x96", "-", regex=False)
            .str.strip()
        )

    print("Data cleaning completed.")
    return df.reset_index(drop=True)


def main() -> None:
    print("=" * 60)
    print("CICIDS2017 Data Preprocessing")
    print("=" * 60)

    csv_files = list_raw_csvs()
    print(f"Found {len(csv_files)} CSV file(s) in data/raw/")

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found. Place the CICIDS2017 files in data/raw/."
        )

    dataframes = []
    encodings = ["utf-8", "cp1252", "latin-1"]

    for file_path in csv_files:
        print(f"Loading {file_path.name}...", end=" ", flush=True)

        for encoding in encodings:
            try:
                dataframe = pd.read_csv(
                    file_path,
                    low_memory=False,
                    encoding=encoding,
                    on_bad_lines="skip",
                )
                dataframes.append(dataframe)
                print(f"{len(dataframe):,} rows")
                break
            except UnicodeDecodeError:
                continue
        else:
            print("Failed to load.")
            warnings.warn(
                f"Could not decode {file_path.name} using supported encodings."
            )

    if not dataframes:
        raise RuntimeError("No data could be loaded. Check the CSV files in data/raw/.")

    print(f"\nCombining {len(dataframes)} dataset(s)...")
    df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined dataset shape: {df.shape}")

    del dataframes

    print("\nCleaning dataset...")
    df = clean_dataframe(df)
    print(f"Cleaned dataset shape: {df.shape}")

    if LABEL_COL not in df.columns:
        raise ValueError(f"Required label column '{LABEL_COL}' was not found.")

    print("\nGenerating class distribution...")
    distribution = df[LABEL_COL].value_counts().reset_index()
    distribution.columns = ["Label", "Count"]
    distribution["Percentage"] = (
        distribution["Count"] / distribution["Count"].sum() * 100
    ).round(3)

    save_table(distribution, "class_distribution.csv")

    print("\nClass distribution (top 15):")
    print(distribution.head(15).to_string(index=False))

    print("\nCreating class distribution plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=distribution.head(15),
        x="Count",
        y="Label",
        ax=ax,
        hue="Label",
        palette="viridis",
        legend=False,
    )

    ax.set_title("CICIDS2017 Class Distribution After Cleaning")
    ax.set_xlabel("Number of Network Flows")
    ax.set_ylabel("Traffic Class")
    fig.tight_layout()

    save_fig(fig, "class_distribution.png")
    plt.close(fig)

    output_path = DATA_PROCESSED / "all_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving cleaned dataset to {output_path}...")
    df.to_csv(output_path, index=False)

    print("\nPreprocessing completed successfully.")
    print(f"Total network flows: {len(df):,}")
    print(f"Total features: {df.shape[1] - 1:,}")
    print(f"Cleaned dataset: {output_path}")


if __name__ == "__main__":
    main()
