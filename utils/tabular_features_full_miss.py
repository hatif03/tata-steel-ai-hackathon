"""Tabular features with missing indicators for all X1–X49 columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.tabular_features import BASE_FEATURES

ALL_MISS_COLS = list(BASE_FEATURES)


def feature_names() -> list[str]:
    return BASE_FEATURES + [f"miss_{c}" for c in ALL_MISS_COLS] + ["row_missing_count"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[BASE_FEATURES].copy()
    for col in ALL_MISS_COLS:
        out[f"miss_{col}"] = df[col].isna().astype(np.int8)
    out["row_missing_count"] = df[BASE_FEATURES].isna().sum(axis=1).astype(np.int16)
    return out[feature_names()]


def to_frame(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df)
