"""Enriched tabular features: base pipeline + ratios, log counts, row aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.tabular_features import (
    BASE_FEATURES,
    MISSING_INDICATOR_COLS,
    build_features as build_base_features,
    feature_names as base_feature_names,
)

CONTINUOUS = [f"X{i}" for i in range(1, 34)]
COUNT = [f"X{i}" for i in range(34, 50)]

# Pairs with strongest univariate PR-AUC signal (EDA + baseline importances).
RATIO_PAIRS = [
    ("X13", "X10"),
    ("X30", "X32"),
    ("X31", "X29"),
    ("X33", "X13"),
    ("X1", "X7"),
    ("X5", "X8"),
    ("X24", "X13"),
    ("X45", "X43"),
]

DERIVED_NAMES = (
    [f"log_{c}" for c in COUNT]
    + [f"ratio_{a}_{b}" for a, b in RATIO_PAIRS]
    + ["cont_mean", "cont_std", "cont_max", "cont_min", "cont_range", "count_sum", "count_nonzero"]
)


def feature_names() -> list[str]:
    return base_feature_names() + DERIVED_NAMES


def _safe_ratio(numer: pd.Series, denom: pd.Series, eps: float = 1e-6) -> pd.Series:
    out = numer / (denom.abs() + eps)
    out[numer.isna() | denom.isna()] = np.nan
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = build_base_features(df)

    for col in COUNT:
        out[f"log_{col}"] = np.log1p(df[col])

    for a, b in RATIO_PAIRS:
        out[f"ratio_{a}_{b}"] = _safe_ratio(df[a], df[b])

    cont = df[CONTINUOUS]
    out["cont_mean"] = cont.mean(axis=1, skipna=True)
    out["cont_std"] = cont.std(axis=1, skipna=True)
    out["cont_max"] = cont.max(axis=1, skipna=True)
    out["cont_min"] = cont.min(axis=1, skipna=True)
    out["cont_range"] = out["cont_max"] - out["cont_min"]

    counts = df[COUNT]
    out["count_sum"] = counts.fillna(0).sum(axis=1)
    out["count_nonzero"] = counts.fillna(0).gt(0).sum(axis=1)

    return out[feature_names()]


def to_frame(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df)
