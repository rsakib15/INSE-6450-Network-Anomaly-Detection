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
from sklearn.metrics import f1_score

from data_loader import prepare_train_val_test
from utils import OUTPUTS, RANDOM_STATE, save_fig, save_table

warnings.filterwarnings("ignore")
MODELS = OUTPUTS / "models"
MODELS.mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_STATE)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    eps = 1e-6
    breaks = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if len(breaks) < 3:
        return 0.0
    e = np.histogram(expected, bins=breaks)[0].astype(float)
    a = np.histogram(actual, bins=breaks)[0].astype(float)
    e = e / (e.sum() + eps) + eps
    a = a / (a.sum() + eps) + eps
    return float(np.sum((a - e) * np.log(a / e)))


def margin(clf, X):
    proba = clf.predict_proba(X)
    part = np.sort(proba, axis=1)
    return part[:, -1] - part[:, -2]


def main():
    print("=" * 60)
    print("DEMO: load → infer → drift → human query → update")
    print("=" * 60)

    data = prepare_train_val_test()
    X_tr = np.asarray(data["X_train"], dtype=np.float64)
    y_tr = np.asarray(data["y_train"])
    X_te = np.asarray(data["X_test"], dtype=np.float64)
    y_te = np.asarray(data["y_test"])

    model_path = MODELS / "rf_baseline.joblib"
    if model_path.exists():
        clf = joblib.load(model_path)["model"]
        print(f"[1] Loaded model: {model_path}")
    else:
        print("[1] No baseline found — training a quick RF on train split...")
        clf = RandomForestClassifier(
            n_estimators=50, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE
        )
        clf.fit(X_tr, y_tr)

    n_demo = min(20000, len(X_te))
    X_batch, y_batch = X_te[:n_demo], y_te[:n_demo]
    t0 = time.perf_counter()
    pred = clf.predict(X_batch)
    infer_s = time.perf_counter() - t0
    f1_clean = float(f1_score(y_batch, pred))
    print(f"[2] Infer on {n_demo} flows: F1={f1_clean:.4f}  time={infer_s:.3f}s")

    ref_std = np.std(X_tr, axis=0) + 1e-8
    X_live = X_batch + 0.4 * ref_std
    n_feat = min(10, X_tr.shape[1])
    mean_psi = float(np.mean([psi(X_tr[:, j], X_live[:, j]) for j in range(n_feat)]))
    drift_alert = mean_psi > 0.2
    print(f"[3] Mean PSI (top {n_feat} feats)={mean_psi:.4f}  alert={drift_alert}")

    pred_drift = clf.predict(X_live)
    f1_before = float(f1_score(y_batch, pred_drift))
    print(f"    F1 on shifted batch BEFORE update: {f1_before:.4f}")

    margins = margin(clf, X_live)
    k = min(500, len(X_live))
    query_idx = np.argsort(margins)[:k]

    X_q, y_q = X_live[query_idx], y_batch[query_idx]
    print(f"[4] Queried {k} uncertain samples for human labelling (oracle sim)")

    buf_n = min(30000, len(X_tr))
    buf_idx = np.random.choice(len(X_tr), size=buf_n, replace=False)
    X_buf = np.vstack([X_tr[buf_idx], X_q])
    y_buf = np.concatenate([y_tr[buf_idx], y_q])

    t0 = time.perf_counter()
    clf2 = RandomForestClassifier(
        n_estimators=50, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE
    )
    clf2.fit(X_buf, y_buf)
    update_s = time.perf_counter() - t0

    pred_after = clf2.predict(X_live)
    f1_after = float(f1_score(y_batch, pred_after))
    f1_clean_after = float(f1_score(y_batch, clf2.predict(X_batch)))
    print(f"[5] Updated model in {update_s:.1f}s")
    print(f"    F1 on shifted batch AFTER update: {f1_after:.4f}")
    print(f"    F1 on clean batch after update:   {f1_clean_after:.4f}")

    joblib.dump(
        {
            "model": clf2,
            "feature_cols": data["feature_cols"],
            "note": "demo_end_to_end_updated",
        },
        MODELS / "rf_demo_updated.joblib",
    )
    print(f"[saved] {MODELS / 'rf_demo_updated.joblib'}")

    summary = pd.DataFrame(
        [
            {
                "stage": "infer_clean",
                "f1": f1_clean,
                "mean_psi": 0.0,
                "drift_alert": False,
                "n_human_labels": 0,
                "seconds": round(infer_s, 4),
            },
            {
                "stage": "shifted_before_update",
                "f1": f1_before,
                "mean_psi": mean_psi,
                "drift_alert": drift_alert,
                "n_human_labels": 0,
                "seconds": 0.0,
            },
            {
                "stage": "shifted_after_update",
                "f1": f1_after,
                "mean_psi": mean_psi,
                "drift_alert": drift_alert,
                "n_human_labels": k,
                "seconds": round(update_s, 2),
            },
        ]
    )
    save_table(summary, "demo_end_to_end_summary.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    stages = ["Clean infer", "Shifted\nbefore update", "Shifted\nafter update"]
    vals = [f1_clean, f1_before, f1_after]
    ax.bar(stages, vals, color=["steelblue", "salmon", "seagreen"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("End-to-end demo: infer → drift → human labels → update")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "demo_end_to_end.png")
    plt.close(fig)
    print("Done. See outputs/tables/demo_end_to_end_summary.csv")


if __name__ == "__main__":
    main()
