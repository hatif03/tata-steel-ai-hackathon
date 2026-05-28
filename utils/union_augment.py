"""GBM-anchored union augment: keep gbm top-K base, add secondary-model exclusives."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from utils.threshold_tuning import apply_top_k

CANARY_COILS = (654, 806, 532, 958, 1187)

DEFAULT_SECONDARY_METHODS = (
    "lightgbm-recall",
    "sklearn-recall",
    "recall-blend",
    "rf-smote-v2",
    "mega-recall-blend",
    "lightgbm-optuna",
    "gbm-mega-blend",
    "catboost-recall",
    "meta-recall-stack",
    "autogluon-recall",
    "xgb-recall",
    "lgb-seedblend-recall",
    "knn-positive-profile",
    "smote-stack-recall",
)

RankingMode = Literal["max", "mean", "weighted"]


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def positive_set(
    df: pd.DataFrame,
    proba_col: str,
    k: int,
    canary_idx: list[int],
) -> set[int]:
    pred, _ = apply_top_k(df[proba_col].values.astype(float), k, force_positive_idx=canary_idx)
    return set(df.loc[pred == 1, "CoilID"].astype(int))


def score_exclusive(
    scores: pd.DataFrame,
    coil: int,
    score_cols: list[str],
    *,
    mode: RankingMode = "max",
    weights: dict[str, float] | None = None,
) -> float:
    values: list[float] = []
    weighted: list[tuple[float, float]] = []
    for col in score_cols:
        if col not in scores.columns:
            continue
        v = float(scores.at[coil, col])
        values.append(v)
        w = 1.0 if weights is None else weights.get(col, 1.0)
        weighted.append((v, w))
    if not values:
        return 0.0
    if mode == "max":
        return float(max(values))
    if mode == "mean":
        return float(np.mean(values))
    total_w = sum(w for _, w in weighted)
    if total_w <= 0:
        return float(np.mean(values))
    return float(sum(v * w for v, w in weighted) / total_w)


def build_augmented(
    test: pd.DataFrame,
    gbm_col: str,
    secondary_methods: tuple[str, ...],
    score_cols: list[str],
    *,
    base_k: int,
    target_k: int,
    secondary_k: int,
    canary_idx: list[int],
    ranking: RankingMode = "max",
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Return (predictions, all positive coils, added exclusives beyond base_k)."""
    base_pred, _ = apply_top_k(test[gbm_col].values.astype(float), base_k, force_positive_idx=canary_idx)
    if target_k <= base_k:
        coils = sorted(test.loc[base_pred == 1, "CoilID"].astype(int).tolist())
        return base_pred, coils, []

    base_coils = set(test.loc[base_pred == 1, "CoilID"].astype(int))
    union_coils: set[int] = set(base_coils)
    for method in secondary_methods:
        if method in test.columns:
            union_coils |= positive_set(test, method, secondary_k, canary_idx)

    candidates = sorted(union_coils - base_coils)
    if not candidates:
        pred, _ = apply_top_k(test[gbm_col].values.astype(float), target_k, force_positive_idx=canary_idx)
        coils = sorted(test.loc[pred == 1, "CoilID"].astype(int).tolist())
        return pred, coils, []

    scores = test.set_index("CoilID")
    ranked = sorted(
        candidates,
        key=lambda c: score_exclusive(scores, c, score_cols, mode=ranking, weights=weights),
        reverse=True,
    )
    selected = set(base_coils)
    added: list[int] = []
    for coil in ranked:
        if len(selected) >= target_k:
            break
        selected.add(int(coil))
        added.append(int(coil))

    if len(selected) < target_k:
        gbm_pred, _ = apply_top_k(test[gbm_col].values.astype(float), target_k, force_positive_idx=canary_idx)
        for coil in test.loc[gbm_pred == 1, "CoilID"].astype(int):
            if len(selected) >= target_k:
                break
            c = int(coil)
            if c not in selected:
                selected.add(c)
                if c not in base_coils:
                    added.append(c)

    pred = np.zeros(len(test), dtype=int)
    coil_to_idx = {int(c): i for i, c in enumerate(test["CoilID"])}
    for coil in selected:
        pred[coil_to_idx[coil]] = 1
    return pred, sorted(selected), added


def load_test_proba(root: Path, method: str) -> pd.DataFrame:
    path = root / f"models/{method}/outputs/latest/predictions/test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)[["CoilID", "proba"]].rename(columns={"proba": method})


def merge_test_probas(
    root: Path,
    *,
    anchor: str = "gbm-recall",
    secondary_methods: tuple[str, ...] = DEFAULT_SECONDARY_METHODS,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge anchor + available secondary probas; skip missing methods."""
    test = load_test_proba(root, anchor)
    loaded: list[str] = []
    for method in secondary_methods:
        try:
            part = load_test_proba(root, method)
            test = test.merge(part, on="CoilID")
            loaded.append(method)
        except FileNotFoundError:
            continue
    return test, loaded
