from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import DATA_PROCESSED, LABEL_COL, EXCLUDE_COLS, RANDOM_STATE


def _clean_label_series(s: pd.Series) -> pd.Series:
    s = s.map(lambda x: str(x).strip() if pd.notna(x) else "")
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan})
    return s


def binary_labels(y: pd.Series) -> np.ndarray:
    y = _clean_label_series(y)
    return (y.str.upper() != "BENIGN").astype(np.int64).values


def select_feature_columns(df: pd.DataFrame) -> list:
    drop = set(EXCLUDE_COLS) | {LABEL_COL}
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        cols.append(c)
    good = []
    for c in cols:
        if c in drop:
            continue
        try:
            pd.to_numeric(df[c], errors="coerce")
            good.append(c)
        except Exception:
            continue
    return good


def load_xy(path, feature_cols=None):
    df = pd.read_csv(path, low_memory=False)
    if LABEL_COL not in df.columns:
        for c in df.columns:
            if str(c).lower() == "label":
                df = df.rename(columns={c: LABEL_COL})
                break
    df[LABEL_COL] = _clean_label_series(df[LABEL_COL])
    df = df[df[LABEL_COL].notna() & (df[LABEL_COL].str.len() > 0)]

    if feature_cols is None:
        feature_cols = select_feature_columns(df)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = binary_labels(df[LABEL_COL])
    return X, y, feature_cols


def prepare_train_val_test(val_size: float = 0.15, max_train_rows=None):
    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing train.csv or test.csv. Run preprocessing.py and split_data.py first."
        )

    X_train_full, y_train_full, feature_cols = load_xy(train_path)
    X_test, y_test, _ = load_xy(test_path, feature_cols=feature_cols)

    if max_train_rows is not None and len(X_train_full) > max_train_rows:
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_train_full), max_train_rows, replace=False
        )
        X_train_full = X_train_full.iloc[idx].reset_index(drop=True)
        y_train_full = y_train_full[idx]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return {
        "X_train": X_tr_s,
        "y_train": y_tr,
        "X_val": X_val_s,
        "y_val": y_val,
        "X_test": X_test_s,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "scaler": scaler,
    }
