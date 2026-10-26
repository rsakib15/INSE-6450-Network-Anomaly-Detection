from __future__ import annotations

from pathlib import Path
import pandas as pd
from utils import TABLES, save_table


def load_test_row(path: Path, model_name: str):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "split" in df.columns:
        row = df[df["split"] == "test"]
        if row.empty:
            row = df.iloc[[-1]]
    else:
        row = df.iloc[[-1]]
    r = row.iloc[0].to_dict()
    r["model"] = model_name
    return r


def main():
    rows = []

    # Prefer full table from updated train_rf_baseline.py
    rf_comp = TABLES / "rf_comparison.csv"
    if rf_comp.exists():
        df = pd.read_csv(rf_comp)
        print("Using rf_comparison.csv")
        save_table(df, "ablation_comparison.csv")
        print(df.to_string(index=False))
        return

    # Fallback: assemble from individual metric files
    for name, fname in [
        ("Random Forest (balanced)", "rf_metrics.csv"),
        ("Random Forest (no class weight)", "rf_metrics_unweighted.csv"),
        ("MLP (weighted)", "mlp_metrics.csv"),
        ("MLP (unweighted)", "mlp_metrics_unweighted.csv"),
    ]:
        r = load_test_row(TABLES / fname, name)
        if r:
            rows.append(r)

    if not rows:
        raise FileNotFoundError("No metrics files found.")

    cols = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "pr_auc",
        "auroc",
    ]
    out = pd.DataFrame(rows)
    keep = [c for c in cols if c in out.columns]
    out = out[keep]
    save_table(out, "ablation_comparison.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
