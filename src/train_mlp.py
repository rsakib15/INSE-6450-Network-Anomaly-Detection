from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from data_loader import prepare_train_val_test
from utils import FIGURES, OUTPUTS, RANDOM_STATE, TABLES, save_fig, save_table

warnings.filterwarnings("ignore")

MODELS = OUTPUTS / "models"
MODELS.mkdir(parents=True, exist_ok=True)


class MLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 2),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int = RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights_from_y(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32)


def metrics_from_logits(logits: np.ndarray, y_true: np.ndarray):
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
    }
    try:
        out["pr_auc"] = float(average_precision_score(y_true, probs))
    except Exception:
        out["pr_auc"] = float("nan")
    try:
        out["auroc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        out["auroc"] = float("nan")
    return out, preds, probs


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).cpu().numpy()
        all_logits.append(logits)
        all_y.append(yb.numpy())
    logits = np.concatenate(all_logits)
    y = np.concatenate(all_y)
    return metrics_from_logits(logits, y)


def plot_learning_curves(history: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_f1"], label="Train F1")
    axes[1].plot(history["epoch"], history["val_f1"], label="Val F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].set_title("F1 vs Epoch")
    axes[1].legend()

    fig.suptitle("MLP Learning Curves (Binary Attack Detection)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "learning_curves.png")
    plt.close(fig)


def plot_confusion(y_true, y_pred, name="confusion_matrix_mlp.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"])
    ax.set_yticklabels(["Benign", "Attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save_fig(fig, name)
    plt.close(fig)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden1", type=int, default=128)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(RANDOM_STATE)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("Milestone 2 – MLP Training (Binary)")
    print(f"Device: {device}")
    print("=" * 60)

    data = prepare_train_val_test(
        val_size=args.val_size,
        max_train_rows=args.max_train_rows,
    )
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_te, y_te = data["X_test"], data["y_test"]

    print(f"Features: {len(data['feature_cols'])}")
    print(f"Train: {len(y_tr):,}  Val: {len(y_val):,}  Test: {len(y_te):,}")
    print(f"Train attack rate: {y_tr.mean():.4f}")

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MLP(
        n_features=X_tr.shape[1],
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
    ).to(device)

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
        print("Class weights: OFF")
    else:
        cw = class_weights_from_y(y_tr).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        print(f"Class weights: {cw.tolist()}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_f1 = -1.0
    best_state = None
    patience_left = args.patience

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            n_batches += 1

        train_m, _, _ = evaluate(model, train_loader, device)
        val_m, _, _ = evaluate(model, val_loader, device)
        train_loss = running_loss / max(n_batches, 1)

        model.eval()
        val_loss_sum, vb = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss_sum += criterion(model(xb), yb).item()
                vb += 1
        val_loss = val_loss_sum / max(vb, 1)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_f1": train_m["f1"],
            "val_f1": val_m["f1"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_pr_auc": val_m["pr_auc"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}  loss={train_loss:.4f}/{val_loss:.4f}  "
            f"F1={train_m['f1']:.4f}/{val_m['f1']:.4f}  "
            f"P={val_m['precision']:.4f} R={val_m['recall']:.4f}"
        )

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    val_m, _, _ = evaluate(model, val_loader, device)
    test_m, test_preds, _ = evaluate(model, test_loader, device)

    print("\n=== Validation ===")
    for k, v in val_m.items():
        print(f"  {k}: {v:.4f}")
    print("=== Test ===")
    for k, v in test_m.items():
        print(f"  {k}: {v:.4f}")

    model_path = MODELS / "mlp_binary.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": X_tr.shape[1],
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "dropout": args.dropout,
            "feature_cols": data["feature_cols"],
            "metrics_test": test_m,
        },
        model_path,
    )
    print(f"[saved] {model_path}")
    joblib.dump(data["scaler"], MODELS / "scaler.joblib")
    print(f"[saved] {MODELS / 'scaler.joblib'}")

    hist_df = pd.DataFrame(history)
    save_table(hist_df, "mlp_history.csv")
    plot_learning_curves(hist_df)
    plot_confusion(y_te, test_preds)

    metrics_rows = []
    for split, m in [("validation", val_m), ("test", test_m)]:
        metrics_rows.append({"split": split, **m})
    save_table(pd.DataFrame(metrics_rows), "mlp_metrics.csv")

    n_params = count_parameters(model)
    disk_mb = model_path.stat().st_size / (1024 * 1024)
    eff = pd.DataFrame(
        [
            {
                "device": str(device),
                "epochs_run": len(history),
                "total_train_seconds": round(train_time, 2),
                "seconds_per_epoch": round(train_time / max(len(history), 1), 2),
                "n_parameters": n_params,
                "model_disk_mb": round(disk_mb, 4),
                "batch_size": args.batch_size,
                "lr": args.lr,
                "class_weights": not args.no_class_weights,
                "best_val_f1": best_val_f1,
            }
        ]
    )
    save_table(eff, "mlp_efficiency.csv")

    cfg = {
        "seed": RANDOM_STATE,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "dropout": args.dropout,
        "patience": args.patience,
        "class_weights": not args.no_class_weights,
        "n_features": int(X_tr.shape[1]),
        "device": str(device),
    }
    cfg_path = MODELS / "mlp_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"[saved] {cfg_path}")
    print("Done.")


if __name__ == "__main__":
    main()
