from __future__ import annotations

import argparse
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_loader import prepare_train_val_test
from utils import OUTPUTS, RANDOM_STATE, TABLES, save_fig, save_table

warnings.filterwarnings("ignore")

MODELS = OUTPUTS / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None:
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            out["pr_auc"] = float("nan")
        try:
            out["auroc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            out["auroc"] = float("nan")
    return out


def plot_confusion(
    y_true, y_pred, name="confusion_matrix_rf.png", title="RF Confusion Matrix (Test)"
):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"])
    ax.set_yticklabels(["Benign", "Attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save_fig(fig, name)
    plt.close(fig)


def plot_learning_curve(curve_df: pd.DataFrame, name="rf_learning_curve.png"):
    """Validation metrics vs number of trees."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        curve_df["n_estimators"], curve_df["val_f1"], marker="o", label="Val F1"
    )
    axes[0].plot(
        curve_df["n_estimators"], curve_df["train_f1"], marker="s", label="Train F1"
    )
    axes[0].set_xlabel("Number of trees")
    axes[0].set_ylabel("F1")
    axes[0].set_title("Learning curve: F1 vs n_estimators")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        curve_df["n_estimators"], curve_df["val_pr_auc"], marker="o", color="darkorange"
    )
    axes[1].set_xlabel("Number of trees")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("Learning curve: Val PR-AUC vs n_estimators")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Random Forest Learning Curves", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, name)
    plt.close(fig)


def plot_comparison_bars(comp_df: pd.DataFrame, name="rf_comparison_bars.png"):
    metrics = [
        c for c in ["precision", "recall", "f1", "pr_auc"] if c in comp_df.columns
    ]
    if not metrics:
        return
    x = np.arange(len(metrics))
    width = 0.15
    models = comp_df["model"].tolist()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, model in enumerate(models):
        vals = [
            float(comp_df.loc[comp_df["model"] == model, m].values[0]) for m in metrics
        ]
        ax.bar(x + i * width, vals, width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (test set)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, name)
    plt.close(fig)


def learning_curve_rf(
    X_tr, y_tr, X_val, y_val, max_trees, max_depth, class_weight, step=10
):
    """Grow forest with warm_start and record metrics every `step` trees."""
    rows = []
    clf = RandomForestClassifier(
        n_estimators=step,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
        warm_start=True,
    )
    n = step
    while n <= max_trees:
        clf.set_params(n_estimators=n)
        clf.fit(X_tr, y_tr)
        tr_pred = clf.predict(X_tr)
        va_pred = clf.predict(X_val)
        va_prob = clf.predict_proba(X_val)[:, 1]
        tr_m = compute_metrics(y_tr, tr_pred)
        va_m = compute_metrics(y_val, va_pred, va_prob)
        rows.append(
            {
                "n_estimators": n,
                "train_f1": tr_m["f1"],
                "val_f1": va_m["f1"],
                "val_precision": va_m["precision"],
                "val_recall": va_m["recall"],
                "val_pr_auc": va_m["pr_auc"],
            }
        )
        print(f"  trees={n:3d}  train_F1={tr_m['f1']:.4f}  val_F1={va_m['f1']:.4f}")
        n += step
    return pd.DataFrame(rows), clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument(
        "--curve-step", type=int, default=10, help="Tree step for learning curve"
    )
    parser.add_argument(
        "--skip-curve", action="store_true", help="Faster: skip learning curve"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Milestone 2 - Random Forest (primary model)")
    print("=" * 60)

    data = prepare_train_val_test(
        val_size=args.val_size,
        max_train_rows=args.max_train_rows,
    )
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_te, y_te = data["X_test"], data["y_test"]

    t0 = time.perf_counter()
    if args.skip_curve:
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        clf.fit(X_tr, y_tr)
        curve_df = None
    else:
        print("Building learning curve (val F1 vs n_estimators)...")
        curve_df, clf = learning_curve_rf(
            X_tr,
            y_tr,
            X_val,
            y_val,
            max_trees=args.n_estimators,
            max_depth=args.max_depth,
            class_weight="balanced",
            step=args.curve_step,
        )
        save_table(curve_df, "rf_learning_curve.csv")
        plot_learning_curve(curve_df)

    train_time = time.perf_counter() - t0

    val_pred = clf.predict(X_val)
    val_prob = clf.predict_proba(X_val)[:, 1]
    test_pred = clf.predict(X_te)
    test_prob = clf.predict_proba(X_te)[:, 1]
    val_m = compute_metrics(y_val, val_pred, val_prob)
    test_m = compute_metrics(y_te, test_pred, test_prob)
    print("\nRF balanced - Validation:", val_m)
    print("RF balanced - Test:", test_m)

    plot_confusion(y_te, test_pred, "confusion_matrix_rf.png", "RF (balanced) - Test")

    joblib.dump(
        {"model": clf, "feature_cols": data["feature_cols"], "scaler": data["scaler"]},
        MODELS / "rf_baseline.joblib",
    )

    save_table(
        pd.DataFrame(
            [
                {"split": "validation", **val_m},
                {"split": "test", **test_m},
            ]
        ),
        "rf_metrics.csv",
    )

    print("\nTraining RF without class_weight (ablation)...")
    clf_none = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight=None,
    )
    clf_none.fit(X_tr, y_tr)
    test_m_none = compute_metrics(
        y_te,
        clf_none.predict(X_te),
        clf_none.predict_proba(X_te)[:, 1],
    )
    print("RF no weight - Test:", test_m_none)

    rows = [
        {"model": "Random Forest (balanced)", **test_m},
        {"model": "Random Forest (no class weight)", **test_m_none},
    ]
    for label, fname in [
        ("MLP (weighted)", "mlp_metrics.csv"),
        ("MLP (unweighted)", "mlp_metrics_unweighted.csv"),
    ]:
        path = TABLES / fname
        if path.exists():
            df = pd.read_csv(path)
            r = df[df["split"] == "test"]
            if not r.empty:
                d = r.iloc[0].to_dict()
                d["model"] = label
                rows.append(d)

    comp = pd.DataFrame(rows)
    keep = [
        c
        for c in ["model", "precision", "recall", "f1", "pr_auc", "accuracy", "auroc"]
        if c in comp.columns
    ]
    comp = comp[keep]
    save_table(comp, "rf_comparison.csv")
    plot_comparison_bars(comp)
    print("\nComparison (test):")
    print(comp.to_string(index=False))

    disk_mb = (MODELS / "rf_baseline.joblib").stat().st_size / (1024 * 1024)
    save_table(
        pd.DataFrame(
            [
                {
                    "total_train_seconds": round(train_time, 2),
                    "n_estimators": args.n_estimators,
                    "max_depth": args.max_depth,
                    "class_weight": "balanced",
                    "model_disk_mb": round(disk_mb, 4),
                    "n_features": int(X_tr.shape[1]),
                }
            ]
        ),
        "rf_efficiency.csv",
    )

    print("\nDone.")
    print(
        "Figures: rf_learning_curve.png, confusion_matrix_rf.png, rf_comparison_bars.png"
    )


if __name__ == "__main__":
    main()
