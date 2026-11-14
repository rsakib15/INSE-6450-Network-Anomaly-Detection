from __future__ import annotations

import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    accuracy_score,
)

from data_loader import prepare_train_val_test
from utils import OUTPUTS, RANDOM_STATE, save_table, save_fig

warnings.filterwarnings("ignore")
MODELS = OUTPUTS / "models"
np.random.seed(RANDOM_STATE)


def metrics(y_true, y_pred, y_prob=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            out["pr_auc"] = float("nan")
    return out


def eval_rf(clf, X, y):
    pred = clf.predict(X)
    prob = clf.predict_proba(X)[:, 1]
    return metrics(y, pred, prob), pred, prob


def add_gaussian_noise(X, sigma):
    return X + np.random.randn(*X.shape) * sigma * (np.std(X, axis=0) + 1e-8)


def mask_features(X, frac):
    X2 = X.copy()
    n_feat = X.shape[1]
    k = max(1, int(frac * n_feat))
    for i in range(len(X2)):
        idx = np.random.choice(n_feat, size=k, replace=False)
        X2[i, idx] = 0.0
    return X2


def out_of_range(X, scale=5.0):
    X2 = X.copy()
    mask = np.random.rand(*X2.shape) < 0.10
    X2[mask] = X2[mask] * scale + np.sign(X2[mask] + 1e-9) * scale
    return X2


def rare_attack_subset(X, y, attack_frac=0.05):
    """Keep all benign; downsample attacks to attack_frac of remaining attacks."""
    atk = np.where(y == 1)[0]
    ben = np.where(y == 0)[0]
    n_keep = max(1, int(len(atk) * attack_frac))
    keep_atk = np.random.choice(atk, size=n_keep, replace=False)
    idx = np.concatenate([ben, keep_atk])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def main():
    print("=" * 60)
    print("Milestone 3 – Robustness evaluation")
    print("=" * 60)

    rf_path = MODELS / "rf_baseline.joblib"
    if not rf_path.exists():
        raise FileNotFoundError("Run train_rf_baseline.py first.")

    pack = joblib.load(rf_path)
    clf = pack["model"]

    data = prepare_train_val_test()
    X = np.asarray(data["X_test"], dtype=np.float64)
    y = np.asarray(data["y_test"])

    rows = []
    m, pred_clean, prob_clean = eval_rf(clf, X, y)
    rows.append({"test": "clean", "model": "RandomForest", **m})
    print("clean", m)

    for sigma in [0.1, 0.25, 0.5, 1.0]:
        Xn = add_gaussian_noise(X, sigma)
        m, _, _ = eval_rf(clf, Xn, y)
        rows.append(
            {"test": f"gaussian_noise_sigma={sigma}", "model": "RandomForest", **m}
        )
        print(f"noise {sigma}", m)

    for frac in [0.1, 0.25, 0.5]:
        Xm = mask_features(X, frac)
        m, _, _ = eval_rf(clf, Xm, y)
        rows.append({"test": f"feature_mask_frac={frac}", "model": "RandomForest", **m})
        print(f"mask {frac}", m)

    Xo = out_of_range(X, scale=5.0)
    m, _, _ = eval_rf(clf, Xo, y)
    rows.append({"test": "out_of_range_x5", "model": "RandomForest", **m})
    print("oor", m)

    Xr, yr = rare_attack_subset(X, y, attack_frac=0.1)
    m, _, _ = eval_rf(clf, Xr, yr)
    rows.append({"test": "rare_attack_eval_10pct", "model": "RandomForest", **m})
    print("rare", m)

    mlp_path = MODELS / "mlp_binary.pt"
    if mlp_path.exists():
        try:
            import torch
            import torch.nn as nn
            from train_mlp import MLP

            ckpt = torch.load(mlp_path, map_location="cpu", weights_only=False)
            mlp = MLP(
                ckpt["n_features"],
                ckpt.get("hidden1", 128),
                ckpt.get("hidden2", 64),
                ckpt.get("dropout", 0.3),
            )
            mlp.load_state_dict(ckpt["state_dict"])
            mlp.eval()

            X_t = torch.tensor(X[:5000], dtype=torch.float32, requires_grad=True)
            y_t = torch.tensor(y[:5000], dtype=torch.long)
            logits = mlp(X_t)
            loss = nn.CrossEntropyLoss()(logits, y_t)
            loss.backward()
            eps = 0.1
            X_adv = (X_t + eps * X_t.grad.sign()).detach()
            with torch.no_grad():
                pred = mlp(X_adv).argmax(1).numpy()
                prob = torch.softmax(mlp(X_adv), 1)[:, 1].numpy()
            m = metrics(y[:5000], pred, prob)
            rows.append({"test": f"mlp_fgsm_eps={eps}", "model": "MLP", **m})
            print("mlp fgsm", m)
        except Exception as e:
            print("MLP adversarial skipped:", e)

    df = pd.DataFrame(rows)
    save_table(df, "m3_robustness_metrics.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    order = ["clean"] + [f"gaussian_noise_sigma={s}" for s in [0.1, 0.25, 0.5, 1.0]]
    sub = df[df["test"].isin(order)].set_index("test").reindex(order)
    ax.plot([0, 0.1, 0.25, 0.5, 1.0], sub["f1"].values, marker="o", label="F1")
    ax.plot([0, 0.1, 0.25, 0.5, 1.0], sub["recall"].values, marker="s", label="Recall")
    ax.set_xlabel("Gaussian noise σ (× feature std)")
    ax.set_ylabel("Score")
    ax.set_title("RF robustness curve: clean → noisy features")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "m3_robustness_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(prob_clean[y == 0], bins=40, alpha=0.6, label="Benign", density=True)
    ax.hist(prob_clean[y == 1], bins=40, alpha=0.6, label="Attack", density=True)
    ax.set_xlabel("Predicted P(attack)")
    ax.set_ylabel("Density")
    ax.set_title("RF confidence histogram (clean test)")
    ax.legend()
    save_fig(fig, "m3_confidence_hist.png")
    plt.close(fig)

    errors = np.where(pred_clean != y)[0]
    conf = np.where(pred_clean == 1, prob_clean, 1 - prob_clean)
    if len(errors) > 0:
        top = errors[np.argsort(-conf[errors])[:10]]
    else:
        top = np.array([], dtype=int)
    cases = []
    for i in top:
        cases.append(
            {
                "index": int(i),
                "true_label": int(y[i]),
                "pred_label": int(pred_clean[i]),
                "p_attack": float(prob_clean[i]),
                "confidence": float(conf[i]),
                "why": "high-confidence misclassification on clean test",
                "mitigation": "raise abstain threshold / human review / retrain on similar flows",
            }
        )
    if len(cases) >= 2:
        cases[0]["resolved_after_adaptation"] = "pending_m3_adaptation"
        cases[1]["resolved_after_adaptation"] = "pending_m3_adaptation"
    save_table(pd.DataFrame(cases), "m3_failure_cases.csv")
    print(f"Saved {len(cases)} failure cases")
    print("Done.")


if __name__ == "__main__":
    main()
