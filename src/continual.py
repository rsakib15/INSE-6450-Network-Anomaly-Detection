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
from utils import OUTPUTS, RANDOM_STATE, save_fig, save_table

warnings.filterwarnings("ignore")
MODELS = OUTPUTS / "models"
MODELS.mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_STATE)


def scores(clf, X, y):
    pred = clf.predict(X)
    prob = clf.predict_proba(X)[:, 1]
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, prob)),
    }


def apply_shift(X, scale, ref_std):
    return X + scale * ref_std


def main():
    print("=" * 60)
    print("Continual learning (RF + replay)")
    print("=" * 60)

    data = prepare_train_val_test()
    X_tr = np.asarray(data["X_train"], dtype=np.float64)
    y_tr = np.asarray(data["y_train"])
    X_te = np.asarray(data["X_test"], dtype=np.float64)
    y_te = np.asarray(data["y_test"])
    ref_std = np.std(X_tr, axis=0) + 1e-8

    base_path = MODELS / "rf_baseline.joblib"
    if base_path.exists():
        clf = joblib.load(base_path)["model"]
        print("Loaded rf_baseline.joblib")
    else:
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE
        )
        clf.fit(X_tr, y_tr)

    buf_size = min(80000, len(X_tr))
    buf_idx = np.random.choice(len(X_tr), size=buf_size, replace=False)
    X_buf, y_buf = X_tr[buf_idx].copy(), y_tr[buf_idx].copy()

    shift_scales = [0.0, 0.15, 0.30, 0.45, 0.60]
    rows = []

    for t, scale in enumerate(shift_scales):
        X_stream = apply_shift(X_te, scale, ref_std)
        before = scores(clf, X_stream, y_te)

        n_new = min(15000, len(X_stream))
        new_idx = np.random.choice(len(X_stream), size=n_new, replace=False)
        X_new = X_stream[new_idx]
        y_new = y_te[new_idx]

        X_buf = np.vstack([X_buf, X_new])
        y_buf = np.concatenate([y_buf, y_new])
        if len(X_buf) > buf_size:
            keep = np.random.choice(len(X_buf), size=buf_size, replace=False)
            X_buf, y_buf = X_buf[keep], y_buf[keep]

        t0 = time.perf_counter()
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE
        )
        clf.fit(X_buf, y_buf)
        update_s = time.perf_counter() - t0

        after = scores(clf, X_stream, y_te)
        clean = scores(clf, X_te, y_te)

        row = {
            "step": t,
            "shift_scale": scale,
            "f1_before_update": before["f1"],
            "f1_after_update": after["f1"],
            "precision_after": after["precision"],
            "recall_after": after["recall"],
            "pr_auc_after": after["pr_auc"],
            "f1_on_clean_test": clean["f1"],
            "update_seconds": round(update_s, 2),
            "buffer_size": len(X_buf),
        }
        rows.append(row)
        print(
            f"step={t} shift={scale:.2f}  F1 before={before['f1']:.4f} "
            f"after={after['f1']:.4f}  clean={clean['f1']:.4f}  update={update_s:.1f}s"
        )

    df = pd.DataFrame(rows)
    save_table(df, "m4_continual_metrics.csv")

    joblib.dump(
        {
            "model": clf,
            "feature_cols": data["feature_cols"],
            "strategy": "replay_full_retrain",
        },
        MODELS / "rf_continual_final.joblib",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    steps = df["step"].values
    axes[0].plot(steps, df["f1_before_update"], "o--", label="F1 before update")
    axes[0].plot(steps, df["f1_after_update"], "s-", label="F1 after update")
    axes[0].plot(steps, df["f1_on_clean_test"], "^-", label="F1 on clean test")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("F1")
    axes[0].set_title("Continual learning trajectory")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(steps, df["update_seconds"], color="steelblue")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Update time (s)")
    axes[1].set_title("Retrain cost per continual update")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Continual learning (RF + replay buffer)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "m4_continual_trajectory.png")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
