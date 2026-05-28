"""Shared vote-union helpers for Phase 11 submissions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.threshold_tuning import apply_top_k
from utils.union_augment import CANARY_COILS

DEFAULT_VOTE_METHODS = (
    "gbm-recall",
    "gbm-recall-safe-fe",
    "lightgbm-recall",
    "sklearn-recall",
    "recall-blend",
    "catboost-recall",
    "catboost-recall-seed42",
    "catboost-recall-seed123",
    "rf-smote-v2",
    "mega-recall-blend",
    "lightgbm-optuna",
    "gbm-mega-blend",
    "meta-recall-stack",
    "autogluon-recall",
    "xgb-recall",
    "lgb-seedblend-recall",
    "knn-positive-profile",
    "smote-stack-recall",
)


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def load_method_proba(root: Path, method: str) -> pd.DataFrame | None:
    path = root / f"models/{method}/outputs/latest/predictions/test_predictions.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)[["CoilID", "proba"]].rename(columns={"proba": method})


def merge_method_probas(root: Path, methods: tuple[str, ...] | list[str]) -> tuple[pd.DataFrame, list[str]]:
    merged = None
    loaded: list[str] = []
    for method in methods:
        part = load_method_proba(root, method)
        if part is None:
            continue
        merged = part if merged is None else merged.merge(part, on="CoilID")
        loaded.append(method)
    if merged is None or len(loaded) < 2:
        raise FileNotFoundError("Need at least 2 model prediction files in vote pool")
    return merged, loaded


def compute_vote_counts(
    merged: pd.DataFrame,
    methods: list[str],
    k: int,
    canary_idx: list[int],
) -> np.ndarray:
    n = len(merged)
    votes = np.zeros(n, dtype=int)
    for method in methods:
        pred, _ = apply_top_k(merged[method].values.astype(float), k, force_positive_idx=canary_idx)
        votes += pred
    return votes


def vote_union_predict(
    votes: np.ndarray,
    min_votes: int,
    canary_idx: list[int],
) -> np.ndarray:
    pred = (votes >= min_votes).astype(int)
    for i in canary_idx:
        pred[i] = 1
    return pred


def vote_distribution(votes: np.ndarray) -> dict[str, int]:
    return {str(v): int((votes == v).sum()) for v in sorted(set(votes.tolist()))}


def write_vote_submission(
    output_dir: Path,
    name: str,
    coil_ids: pd.Series,
    pred: np.ndarray,
    meta: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "submission.csv"
    pd.DataFrame({"CoilID": coil_ids, "Y": pred}).to_csv(csv_path, index=False)
    (output_dir / "meta.json").write_text(json.dumps({"strategy": name, **meta}, indent=2), encoding="utf-8")
    return csv_path


def load_oof_weights(root: Path, methods: list[str]) -> dict[str, float]:
    """OOF accuracy @ top-K from metrics.json as vote weights (fallback equal)."""
    weights: dict[str, float] = {}
    for method in methods:
        metrics_path = root / f"models/{method}/outputs/latest/metrics.json"
        w = 1.0
        if metrics_path.is_file():
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            w = float(data.get("oof_accuracy", data.get("oof_pr_auc", 1.0)))
        weights[method] = max(w, 1e-6)
    total = sum(weights.values())
    return {m: weights[m] / total for m in methods}
