from __future__ import annotations

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
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)

from data_loader import prepare_train_val_test
from utils import OUTPUTS, RANDOM_STATE, save_table, save_fig

warnings.filterwarnings("ignore")
MODELS = OUTPUTS / "models"
MODELS.mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_STATE)


def eval_scores(clf, X, y):
    pred = clf.predict(X)
    prob = clf.predict_proba(X)[:, 1]
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, prob)),
    }


def main():
    print("=" * 60)
    print("Milestone 3 – Adaptation under drift")
    print("=" * 60)

    data = prepare_train_val_test()
    X_tr = np.asarray(data["X_train"], dtype=np.float64)
    y_tr = np.asarray(data["y_train"])
    X_te = np.asarray(data["X_test"], dtype=np.float64)
    y_te = np.asarray(data["y_test"])

    base_path = MODELS / "rf_baseline.joblib"
    if not base_path.exists():
        raise FileNotFoundError("Train RF first: python src/train_rf_baseline.py")
    base = joblib.load(base_path)["model"]

    X_drift = X_te.copy()
    shift = 0.75 * np.std(X_tr, axis=0)
    X_drift = X_drift + shift

    atk = np.where(y_te == 1)[0]
    ben = np.where(y_te == 0)[0]
    keep_atk = np.random.choice(atk, size=max(1, len(atk) // 3), replace=False)
    idx = np.concatenate([ben, keep_atk])
    X_drift_eval, y_drift_eval = X_drift[idx], y_te[idx]

    before = eval_scores(base, X_drift_eval, y_drift_eval)
    clean = eval_scores(base, X_te, y_te)
    print("Clean test:", clean)
    print("Before on drifted:", before)

    n_aug = min(50000, len(X_tr))
    sel = np.random.choice(len(X_tr), size=n_aug, replace=False)
    X_aug = X_tr[sel] + shift
    y_aug = y_tr[sel]
    X_new = np.vstack([X_tr, X_aug])
    y_new = np.concatenate([y_tr, y_aug])

    t0 = time.perf_counter()
    adapted = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight=None,
    )
    adapted.fit(X_new, y_new)
    train_time = time.perf_counter() - t0

    after = eval_scores(adapted, X_drift_eval, y_drift_eval)
    after_clean = eval_scores(adapted, X_te, y_te)
    print("After on drifted:", after)
    print("After on clean:", after_clean)

    joblib.dump(
        {
            "model": adapted,
            "feature_cols": data["feature_cols"],
            "note": "adapted_under_shift",
        },
        MODELS / "rf_adapted.joblib",
    )

    rows = [
        {"stage": "baseline_clean_test", **clean},
        {"stage": "baseline_on_drifted", **before},
        {"stage": "adapted_on_drifted", **after},
        {"stage": "adapted_on_clean_test", **after_clean},
        {
            "stage": "adapt_train_seconds",
            "precision": train_time,
            "recall": 0,
            "f1": 0,
            "pr_auc": 0,
        },
    ]
    save_table(pd.DataFrame(rows), "m3_adaptation_metrics.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["precision", "recall", "f1", "pr_auc"]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [before[k] for k in labels], 0.4, label="Before (baseline RF)")
    ax.bar(x + 0.2, [after[k] for k in labels], 0.4, label="After (adapted RF)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Adaptation: metrics on drifted evaluation set")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, "m3_adaptation_before_after.png")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
