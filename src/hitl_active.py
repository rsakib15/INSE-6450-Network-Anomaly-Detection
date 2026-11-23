from __future__ import annotations

import time
import warnings

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
from sklearn.model_selection import train_test_split

from data_loader import prepare_train_val_test
from utils import RANDOM_STATE, save_fig, save_table

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)


def metrics(y_true, pred, prob):
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, prob)),
    }


def margin_scores(clf, X):
    proba = clf.predict_proba(X)
    if proba.shape[1] == 1:
        return np.zeros(len(X))
    part = np.sort(proba, axis=1)
    return part[:, -1] - part[:, -2]


def main():
    print("=" * 60)
    print("Active learning / HITL simulation")
    print("=" * 60)

    data = prepare_train_val_test()
    X = np.asarray(data["X_train"], dtype=np.float64)
    y = np.asarray(data["y_train"])
    X_te = np.asarray(data["X_test"], dtype=np.float64)
    y_te = np.asarray(data["y_test"])

    X_lab, X_pool, y_lab, y_pool = train_test_split(
        X, y, test_size=0.90, random_state=RANDOM_STATE, stratify=y
    )
    max_pool = min(100000, len(X_pool))
    if len(X_pool) > max_pool:
        idx = np.random.choice(len(X_pool), size=max_pool, replace=False)
        X_pool, y_pool = X_pool[idx], y_pool[idx]

    seed_n = min(2000, len(X_lab))
    X_seed, y_seed = X_lab[:seed_n], y_lab[:seed_n]

    query_size = 500
    n_rounds = 8
    rows = []

    X_L, y_L = X_seed.copy(), y_seed.copy()
    X_U, y_U = X_pool.copy(), y_pool.copy()

    for r in range(n_rounds + 1):
        t0 = time.perf_counter()
        clf = RandomForestClassifier(
            n_estimators=80, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE
        )
        clf.fit(X_L, y_L)
        fit_s = time.perf_counter() - t0

        pred = clf.predict(X_te)
        prob = clf.predict_proba(X_te)[:, 1]
        m = metrics(y_te, pred, prob)
        rows.append(
            {
                "round": r,
                "n_labelled": len(X_L),
                "n_pool": len(X_U),
                "test_f1": m["f1"],
                "test_precision": m["precision"],
                "test_recall": m["recall"],
                "test_pr_auc": m["pr_auc"],
                "fit_seconds": round(fit_s, 2),
                "labels_queried_this_round": 0 if r == 0 else query_size,
            }
        )
        print(
            f"round={r} labelled={len(X_L):,}  test_F1={m['f1']:.4f}  fit={fit_s:.1f}s"
        )

        if r == n_rounds or len(X_U) < query_size:
            break

        margins = margin_scores(clf, X_U)
        order = np.argsort(margins)
        pick = order[:query_size]

        X_q, y_q = X_U[pick], y_U[pick]
        X_L = np.vstack([X_L, X_q])
        y_L = np.concatenate([y_L, y_q])
        mask = np.ones(len(X_U), dtype=bool)
        mask[pick] = False
        X_U, y_U = X_U[mask], y_U[mask]

    df = pd.DataFrame(rows)
    save_table(df, "m4_active_learning_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["n_labelled"], df["test_f1"], "o-", label="Test F1")
    axes[0].plot(df["n_labelled"], df["test_pr_auc"], "s--", label="Test PR-AUC")
    axes[0].set_xlabel("Number of labelled samples")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Active learning curve (uncertainty / margin)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["round"], df["fit_seconds"], "o-", color="darkorange")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Fit time (s)")
    axes[1].set_title("Retrain cost per active-learning round")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("HITL / active learning simulation", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "m4_active_learning_curve.png")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
