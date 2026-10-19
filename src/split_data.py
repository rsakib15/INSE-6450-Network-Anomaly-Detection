import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from utils import DATA_PROCESSED, TABLES, LABEL_COL, RANDOM_STATE, save_table, save_fig

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def plot_split_figure(y_train, y_test, test_size: float):
    train_benign = int((y_train.str.upper() == "BENIGN").sum())
    train_attack = int((y_train.str.upper() != "BENIGN").sum())
    test_benign = int((y_test.str.upper() == "BENIGN").sum())
    test_attack = int((y_test.str.upper() != "BENIGN").sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax0 = axes[0]
    sizes = [len(y_train), len(y_test)]
    labels = [
        f"Train\n{len(y_train):,} rows\n({100 * (1 - test_size):.0f}%)",
        f"Test\n{len(y_test):,} rows\n({100 * test_size:.0f}%)",
    ]
    colors = ["#2c7bb6", "#d7191c"]
    wedges, texts, autotexts = ax0.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="",
        startangle=90,
        explode=(0.02, 0.02),
        textprops={"fontsize": 10},
    )
    ax0.set_title("Train / Test Split Size", fontsize=12, fontweight="bold")

    ax1 = axes[1]
    categories = ["Train", "Test"]
    benign_counts = [train_benign, test_benign]
    attack_counts = [train_attack, test_attack]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2, benign_counts, width, label="Benign", color="#2ca25f"
    )
    bars2 = ax1.bar(
        x + width / 2, attack_counts, width, label="Attack", color="#e34a33"
    )

    ax1.set_ylabel("Number of flows")
    ax1.set_title(
        "Benign vs Attack by Split (Stratified)", fontsize=12, fontweight="bold"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax1.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(
        "CICIDS2017 - Stratified 80/20 Train-Test Split",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "train_test_split.png")
    plt.close(fig)


def main(test_size: float = 0.20):
    print("=" * 60)
    print("Train / Test Split")
    print("=" * 60)

    data_path = DATA_PROCESSED / "all_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run preprocessing.py first.")

    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Full dataset shape: {df.shape}")

    # Ensure Label column exists
    if LABEL_COL not in df.columns:
        for c in df.columns:
            if str(c).lower() == "label":
                df = df.rename(columns={c: LABEL_COL})
                break

    if LABEL_COL not in df.columns:
        raise KeyError(f"Label column '{LABEL_COL}' not found in the data.")

    df[LABEL_COL] = df[LABEL_COL].map(lambda x: str(x).strip() if pd.notna(x) else "")
    df = df[df[LABEL_COL].str.len() > 0]
    df = df[~df[LABEL_COL].str.lower().isin(["nan", "none", "null"])]

    y = df[LABEL_COL]
    X = df.drop(columns=[LABEL_COL])

    print(
        f"\nCreating stratified {int((1 - test_size) * 100)}/"
        f"{int(test_size * 100)} train/test split ..."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train = X_train.copy()
    train[LABEL_COL] = y_train.values
    test = X_test.copy()
    test[LABEL_COL] = y_test.values

    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"

    print(f"Saving train → {train_path}")
    train.to_csv(train_path, index=False)
    print(f"Saving test  → {test_path}")
    test.to_csv(test_path, index=False)

    # Summary table
    summary = pd.DataFrame(
        {
            "Split": ["Train", "Test", "Total"],
            "Rows": [len(train), len(test), len(train) + len(test)],
            "Share": [
                f"{100 * (1 - test_size):.0f}%",
                f"{100 * test_size:.0f}%",
                "100%",
            ],
            "Benign": [
                int((y_train.str.upper() == "BENIGN").sum()),
                int((y_test.str.upper() == "BENIGN").sum()),
                int((y.str.upper() == "BENIGN").sum()),
            ],
            "Attack": [
                int((y_train.str.upper() != "BENIGN").sum()),
                int((y_test.str.upper() != "BENIGN").sum()),
                int((y.str.upper() != "BENIGN").sum()),
            ],
        }
    )
    save_table(summary, "split_summary.csv")

    print("\nCreating split figure ...")
    plot_split_figure(y_train, y_test, test_size)

    print("\nSplit summary:")
    print(summary.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
