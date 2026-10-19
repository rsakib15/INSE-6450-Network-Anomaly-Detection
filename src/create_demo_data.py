import numpy as np
import pandas as pd
from pathlib import Path
from utils import DATA_RAW, RANDOM_STATE

rng = np.random.RandomState(RANDOM_STATE)

N = 20_000


def make_flows(n, label, attack_intensity=1.0):
    base = {
        "Flow Duration": rng.exponential(1e5, n) * attack_intensity,
        "Total Fwd Packets": rng.poisson(10 * attack_intensity, n),
        "Total Backward Packets": rng.poisson(8, n),
        "Total Length of Fwd Packets": rng.exponential(500, n) * attack_intensity,
        "Total Length of Bwd Packets": rng.exponential(400, n),
        "Fwd Packet Length Mean": rng.normal(200, 50, n).clip(0),
        "Bwd Packet Length Mean": rng.normal(180, 40, n).clip(0),
        "Fwd Packet Length Std": rng.exponential(30, n),
        "Bwd Packet Length Std": rng.exponential(25, n) * attack_intensity,
        "Flow Bytes/s": rng.exponential(1e4, n) * attack_intensity,
        "Flow Packets/s": rng.exponential(50, n),
        "Flow IAT Mean": rng.exponential(1e4, n),
        "Flow IAT Std": rng.exponential(5e3, n),
        "Flow IAT Max": rng.exponential(5e4, n),
        "Flow IAT Min": rng.exponential(100, n),
        "Fwd IAT Total": rng.exponential(1e5, n),
        "Bwd Packet Length Max": rng.normal(500, 100, n).clip(0),
        "Fwd Packet Length Max": rng.normal(600, 120, n).clip(0),
        "Fwd Packet Length Min": rng.normal(40, 10, n).clip(0),
        "Label": [label] * n,
    }
    return pd.DataFrame(base)


print("Generating synthetic CICIDS2017-like data ...")

benign = make_flows(int(N * 0.83), "BENIGN", 1.0)
hulk = make_flows(int(N * 0.08), "DoS Hulk", 3.0)
pscan = make_flows(int(N * 0.05), "PortScan", 0.3)
ddos = make_flows(int(N * 0.02), "DDoS", 4.0)
other = make_flows(int(N * 0.02), "FTP-Patator", 1.5)

df = pd.concat([benign, hulk, pscan, ddos, other], ignore_index=True)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

df["Flow ID"] = [f"flow_{i}" for i in range(len(df))]
df["Source IP"] = "192.168.1." + pd.Series(rng.randint(1, 255, len(df))).astype(str)
df["Destination Port"] = rng.choice([80, 443, 22, 21, 53], len(df))
df["Timestamp"] = pd.date_range("2017-07-03", periods=len(df), freq="s")

out = DATA_RAW / "demo_cicids2017.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df):,} synthetic rows → {out}")
print("You can now run:  python src/preprocessing.py")
