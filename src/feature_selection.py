import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (no GUI / no tkinter)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from utils import (
    DATA_PROCESSED,
    ATTACKS_DIR,
    FIGURES,
    TABLES,
    save_table,
    save_fig,
    LABEL_COL,
    EXCLUDE_COLS,
    RANDOM_STATE,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def prepare_xy(df: pd.DataFrame, label_col: str = LABEL_COL):
    drop_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    drop_cols.append(label_col)
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))
    return X, y, le


def rf_importance(X: pd.DataFrame, y: np.ndarray, n_estimators: int = 30):
    max_rows = 80_000
    if len(X) > max_rows:
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X), max_rows, replace=False
        )
        X = X.iloc[idx]
        y = y[idx]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        max_features="sqrt",
    )
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns)
    return imp.sort_values(ascending=False)


def plot_importance(imp: pd.Series, title: str, filename: str, top_n: int = 15):
    fig, ax = plt.subplots(figsize=(9, 6))
    top = imp.head(top_n)
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="magma")
    ax.set_title(title)
    ax.set_xlabel("Importance weight")
    save_fig(fig, filename)
    plt.close(fig)
    plt.close("all")


def create_attack_files(df: pd.DataFrame):
    df = df.copy()
    df[LABEL_COL] = df[LABEL_COL].map(lambda x: str(x).strip() if pd.notna(x) else "")
    df = df[df[LABEL_COL].str.len() > 0]
    df = df[~df[LABEL_COL].str.lower().isin(["nan", "none", "null"])]

    is_benign = df[LABEL_COL].str.upper() == "BENIGN"
    benign = df[is_benign]
    attacks = df[~is_benign]

    attack_labels = sorted({str(x) for x in attacks[LABEL_COL].tolist()})
    print(f"\nCreating balanced files for {len(attack_labels)} attack types ...")
    print(f"  Attack labels: {attack_labels}")

    if len(benign) == 0:
        print("WARNING: No BENIGN rows found. Skipping balanced file creation.")
        return []

    summary = []
    for label in attack_labels:
        atk = attacks[attacks[LABEL_COL] == label]
        n_atk = len(atk)
        if n_atk == 0:
            continue

        n_ben = int(n_atk * 0.7 / 0.3)
        n_ben = min(n_ben, len(benign))

        ben_sample = benign.sample(n=n_ben, random_state=RANDOM_STATE)
        balanced = pd.concat([atk, ben_sample], ignore_index=True)
        balanced = balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(
            drop=True
        )

        safe_name = (
            str(label)
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("–", "_")
            .replace(".", "_")
        )
        out = ATTACKS_DIR / f"{safe_name}.csv"
        balanced.to_csv(out, index=False)

        summary.append(
            {
                "Attack": label,
                "Attack_rows": n_atk,
                "Benign_rows": n_ben,
                "Total": len(balanced),
                "File": out.name,
            }
        )
        print(f"  {label:40s} → {out.name}  ({len(balanced):,} rows)")

    save_table(pd.DataFrame(summary), "attack_files_summary.csv")
    return summary


def main():
    print("=" * 60)
    print("Feature Selection with Random Forest Regressor")
    print("=" * 60)

    data_path = DATA_PROCESSED / "all_data.csv"
    if not data_path.exists():
        raise FileNotFoundError("Run preprocessing.py first.")

    print("Loading cleaned data (this may take a minute) ...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Shape: {df.shape}")

    if LABEL_COL not in df.columns:
        for c in df.columns:
            if str(c).lower() == "label":
                df.rename(columns={c: LABEL_COL}, inplace=True)
                break

    df[LABEL_COL] = df[LABEL_COL].map(lambda x: str(x).strip() if pd.notna(x) else "")
    df = df[df[LABEL_COL].str.len() > 0]
    df = df[~df[LABEL_COL].str.lower().isin(["nan", "none", "null"])]

    print("\n[1/3] Computing global feature importance (Attack vs Benign) ...")
    df_bin = df.copy()
    df_bin[LABEL_COL] = np.where(
        df_bin[LABEL_COL].str.upper() == "BENIGN", "BENIGN", "Attack"
    )

    X, y, _ = prepare_xy(df_bin)
    imp_global = rf_importance(X, y)
    save_table(
        imp_global.reset_index().rename(columns={"index": "feature", 0: "importance"}),
        "feature_importance_all.csv",
    )
    plot_importance(
        imp_global,
        "Global Feature Importance (Attack vs Benign)",
        "feature_importance_global.png",
    )

    print("Top 10 global features:")
    print(imp_global.head(10))

    print("\n[2/3] Creating per-attack balanced files ...")
    create_attack_files(df)

    print("\n[3/3] Computing per-attack feature importance ...")
    all_attack_imp = []

    for f in sorted(ATTACKS_DIR.glob("*.csv")):
        attack_name = f.stem
        print(f"  Processing {attack_name} ...", end=" ", flush=True)
        atk_df = pd.read_csv(f, low_memory=False)

        X, y, _ = prepare_xy(atk_df)
        if X.shape[1] == 0:
            print("skipped (no numeric features)")
            continue

        imp = rf_importance(X, y, n_estimators=30)
        top4 = imp.head(4)
        for feat, weight in top4.items():
            all_attack_imp.append(
                {"Attack": attack_name, "Feature": feat, "Importance": weight}
            )
        print("ok")

        plot_importance(
            imp,
            f"Feature Importance – {attack_name}",
            f"feature_importance_{attack_name}.png",
            top_n=10,
        )

    imp_df = pd.DataFrame(all_attack_imp)
    save_table(imp_df, "feature_importance_per_attack.csv")
    print("\nPer-attack top features saved.")

    print("\nFeature selection finished successfully.")
    print(f"Artefacts are in:\n  {TABLES}\n  {FIGURES}\n  {ATTACKS_DIR}")


if __name__ == "__main__":
    main()
