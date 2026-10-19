import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ATTACKS_DIR = DATA_PROCESSED / "attacks"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

for d in [DATA_RAW, DATA_PROCESSED, ATTACKS_DIR, FIGURES, TABLES]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

EXCLUDE_COLS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "External IP",
]

LABEL_COL = "Label"


def list_raw_csvs():
    files = sorted(DATA_RAW.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_RAW}.\n"
            "Please download CICIDS2017 CSV files and place them there."
        )
    return files


def save_table(df, name: str):
    path = TABLES / name
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def save_fig(fig, name: str, dpi=150):
    path = FIGURES / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {path}")
