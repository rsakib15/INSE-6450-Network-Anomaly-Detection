from __future__ import annotations

import os
import time

import joblib
import numpy as np
import pandas as pd
import psutil

from data_loader import prepare_train_val_test
from utils import OUTPUTS, TABLES

MODELS = OUTPUTS / "models"
MODELS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

proc = psutil.Process(os.getpid())


def rss_mb() -> float:
    return proc.memory_info().rss / (1024**2)


def measure_latency(clf, X, batch_size: int, warmup: int = 30, runs: int = 200):
    n = len(X)
    for i in range(warmup):
        s = (i * batch_size) % max(n - batch_size, 1)
        clf.predict(X[s : s + batch_size])

    times = []
    n_samples = 0
    for i in range(runs):
        s = (i * batch_size) % max(n - batch_size, 1)
        xb = X[s : s + batch_size]
        t0 = time.perf_counter()
        clf.predict(xb)
        dt = time.perf_counter() - t0
        times.append(dt)
        n_samples += batch_size

    times = np.asarray(times, dtype=np.float64)
    per_sample_ms = times / batch_size * 1000.0
    throughput = n_samples / times.sum() if times.sum() > 0 else float("nan")
    return {
        "p50_ms_per_sample": float(np.percentile(per_sample_ms, 50)),
        "p90_ms_per_sample": float(np.percentile(per_sample_ms, 90)),
        "throughput_samples_per_s": float(throughput),
    }


def main():
    path = MODELS / "rf_baseline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train_rf_baseline.py first.")

    disk_mb = path.stat().st_size / (1024**2)

    baseline = rss_mb()
    pack = joblib.load(path)
    clf = pack["model"]
    after_load = rss_mb()

    data = prepare_train_val_test()
    X = data["X_test"][:10000]

    rows = []
    for bs in [1, 256]:
        # score a few times so RSS reflects inference path
        for _ in range(50):
            clf.predict(X[0:bs])
        after = rss_mb()
        lat = measure_latency(clf, X, batch_size=bs)
        rows.append(
            {
                "batch_size": bs,
                "model_disk_mb": round(disk_mb, 4),
                "rss_before_load_mb": round(baseline, 2),
                "rss_after_load_mb": round(after_load, 2),
                "model_load_delta_mb": round(after_load - baseline, 2),
                "rss_after_inference_mb": round(after, 2),
                "p50_ms_per_sample": round(lat["p50_ms_per_sample"], 6),
                "p90_ms_per_sample": round(lat["p90_ms_per_sample"], 6),
                "throughput_samples_per_s": round(lat["throughput_samples_per_s"], 2),
            }
        )

    df = pd.DataFrame(rows)
    out = TABLES / "rf_inference_efficiency.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
