"""Shared tabular features: raw X1–X49 with missingness indicators (NaN retained)."""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = [f"X{i}" for i in range(1, 50)]

MISSING_INDICATOR_COLS = [
    "X15",
    "X42",
    "X48",
    "X10",
    "X16",
    "X23",
    "X24",
    "X25",
    "X26",
    "X27",
]


def feature_names() -> list[str]:
    return BASE_FEATURES + [f"miss_{c}" for c in MISSING_INDICATOR_COLS] + ["row_missing_count"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[BASE_FEATURES].copy()
    for col in MISSING_INDICATOR_COLS:
        out[f"miss_{col}"] = df[col].isna().astype(np.int8)
    out["row_missing_count"] = df[BASE_FEATURES].isna().sum(axis=1).astype(np.int16)
    return out[feature_names()]


def to_frame(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df)
