from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import prepare_train_val_test
from utils import RANDOM_STATE, save_table, save_fig

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    eps = 1e-6
    quantiles = np.linspace(0, 100, bins + 1)
    breaks = np.unique(np.percentile(expected, quantiles))
    if len(breaks) < 3:
        return 0.0
    e_counts = np.histogram(expected, bins=breaks)[0].astype(float)
    a_counts = np.histogram(actual, bins=breaks)[0].astype(float)
    e_perc = e_counts / (e_counts.sum() + eps) + eps
    a_perc = a_counts / (a_counts.sum() + eps) + eps
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def main():
    print("=" * 60)
    print("Milestone 3 – Monitoring / drift simulation")
    print("=" * 60)

    data = prepare_train_val_test()
    X_ref = np.asarray(data["X_train"], dtype=np.float64)
    X_live = np.asarray(data["X_test"], dtype=np.float64)
    feat_names = data["feature_cols"]

    n_feat = min(12, X_ref.shape[1])  # top features for dashboard
    psi_rows = []
    for j in range(n_feat):
        p = psi(X_ref[:, j], X_live[:, j])
        psi_rows.append(
            {
                "feature": feat_names[j] if j < len(feat_names) else f"f{j}",
                "psi_test_vs_train": p,
                "batch": "test_natural",
            }
        )

    X_shift = X_live.copy()
    X_shift[:, :n_feat] = X_shift[:, :n_feat] + 0.5 * np.std(X_ref[:, :n_feat], axis=0)
    for j in range(n_feat):
        p = psi(X_ref[:, j], X_shift[:, j])
        psi_rows.append(
            {
                "feature": feat_names[j] if j < len(feat_names) else f"f{j}",
                "psi_test_vs_train": p,
                "batch": "simulated_shift",
            }
        )

    psi_df = pd.DataFrame(psi_rows)
    save_table(psi_df, "m3_drift_psi.csv")

    logs = []
    batch_size = 5000
    for b, start in enumerate(range(0, min(len(X_live), 25000), batch_size)):
        xb = X_live[start : start + batch_size]
        null_rate = float(np.isnan(xb).mean()) if np.isnan(xb).any() else 0.0
        mean_psi = float(np.mean([psi(X_ref[:, j], xb[:, j]) for j in range(n_feat)]))
        alert = mean_psi > 0.2
        logs.append(
            {
                "batch_id": b,
                "n_rows": len(xb),
                "null_rate": null_rate,
                "mean_psi_top_features": mean_psi,
                "drift_alert": alert,
                "schema_ok": True,
            }
        )

    logs.append(
        {
            "batch_id": 99,
            "n_rows": batch_size,
            "null_rate": 0.15,
            "mean_psi_top_features": 0.45,
            "drift_alert": True,
            "schema_ok": False,
        }
    )
    log_df = pd.DataFrame(logs)
    save_table(log_df, "m3_monitoring_log.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    natural = psi_df[psi_df["batch"] == "test_natural"].sort_values("psi_test_vs_train")
    axes[0].barh(natural["feature"], natural["psi_test_vs_train"], color="steelblue")
    axes[0].axvline(0.1, color="orange", ls="--", label="PSI 0.1")
    axes[0].axvline(0.2, color="red", ls="--", label="PSI 0.2")
    axes[0].set_xlabel("PSI")
    axes[0].set_title("Feature PSI (test vs train)")
    axes[0].legend(fontsize=8)

    axes[1].plot(log_df["batch_id"], log_df["mean_psi_top_features"], marker="o")
    axes[1].axhline(0.2, color="red", ls="--", label="alert threshold")
    axes[1].set_xlabel("Batch id")
    axes[1].set_ylabel("Mean PSI")
    axes[1].set_title("Streaming drift monitor (simulated batches)")
    axes[1].legend()
    fig.suptitle("Milestone 3 – Monitoring dashboard", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "m3_drift_dashboard.png")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
