"""Safe FE: gbm-recall features + ratio_X13_X7 (passed canary guard probe)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.tabular_features import build_features as base_build

BASE_FEATURES = [f"X{i}" for i in range(1, 50)]
EXTRA = ("ratio_X13_X7",)


def feature_names() -> list[str]:
    return list(base_build(pd.DataFrame(columns=BASE_FEATURES)).columns) + list(EXTRA)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = base_build(df)
    out["ratio_X13_X7"] = df["X13"] / df["X7"].replace(0, np.nan)
    return out


def to_frame(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df)
