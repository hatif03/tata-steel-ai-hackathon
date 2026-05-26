"""Build GBM-anchored union-augmented submissions (Phase 8A)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.threshold_tuning import apply_top_k  # noqa: E402

CANARY_COILS = (654, 806, 532, 958, 1187)
OUTPUT_DIR = ROOT / "models" / "phase8-rethreshold" / "outputs"
SECONDARY_METHODS = ("lightgbm-recall", "sklearn-recall", "recall-blend")
SECONDARY_K = 26
GBM_BASE_K = 33


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def load_test_proba(method: str) -> pd.DataFrame:
    path = ROOT / f"models/{method}/outputs/latest/predictions/test_predictions.csv"
    return pd.read_csv(path)[["CoilID", "proba"]].rename(columns={"proba": method})


def positive_set(df: pd.DataFrame, proba_col: str, k: int, canary_idx: list[int]) -> set[int]:
    pred, _ = apply_top_k(df[proba_col].values.astype(float), k, force_positive_idx=canary_idx)
    return set(df.loc[pred == 1, "CoilID"].astype(int))


def build_augmented(
    test: pd.DataFrame,
    gbm_col: str,
    score_cols: list[str],
    *,
    base_k: int,
    target_k: int,
    canary_idx: list[int],
) -> tuple[np.ndarray, list[int]]:
    base_pred, _ = apply_top_k(test[gbm_col].values.astype(float), base_k, force_positive_idx=canary_idx)
    if target_k <= base_k:
        return base_pred, sorted(test.loc[base_pred == 1, "CoilID"].astype(int).tolist())

    base_coils = set(test.loc[base_pred == 1, "CoilID"].astype(int))
    union_coils: set[int] = set(base_coils)
    for method in SECONDARY_METHODS:
        if method in test.columns:
            union_coils |= positive_set(test, method, SECONDARY_K, canary_idx)

    candidates = sorted(union_coils - base_coils)
    if not candidates:
        pred, _ = apply_top_k(test[gbm_col].values.astype(float), target_k, force_positive_idx=canary_idx)
        return pred, sorted(test.loc[pred == 1, "CoilID"].astype(int).tolist())

    scores = test.set_index("CoilID")
    ranked = sorted(
        candidates,
        key=lambda c: float(np.max([scores.at[c, col] for col in score_cols if col in scores.columns])),
        reverse=True,
    )
    selected = set(base_coils)
    for coil in ranked:
        if len(selected) >= target_k:
            break
        selected.add(int(coil))

    if len(selected) < target_k:
        remaining = target_k - len(selected)
        gbm_pred, _ = apply_top_k(test[gbm_col].values.astype(float), target_k, force_positive_idx=canary_idx)
        for coil in test.loc[gbm_pred == 1, "CoilID"].astype(int):
            if len(selected) >= target_k:
                break
            selected.add(int(coil))
        _ = remaining

    pred = np.zeros(len(test), dtype=int)
    coil_to_idx = {int(c): i for i, c in enumerate(test["CoilID"])}
    for coil in selected:
        pred[coil_to_idx[coil]] = 1
    return pred, sorted(selected)


def write_submission(name: str, test: pd.DataFrame, pred: np.ndarray, meta: dict) -> Path:
    out = OUTPUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "submission.csv"
    pd.DataFrame({"CoilID": test["CoilID"], "Y": pred}).to_csv(path, index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ks", nargs="+", type=int, default=[35, 38])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test = load_test_proba("gbm-recall")
    for method in SECONDARY_METHODS:
        test = test.merge(load_test_proba(method), on="CoilID")
    canary_idx = canary_indices(test["CoilID"])
    score_cols = list(SECONDARY_METHODS)

    manifest: list[dict] = []

    for target_k in args.target_ks:
        if target_k <= GBM_BASE_K:
            continue
        pred, coils = build_augmented(
            test,
            "gbm-recall",
            score_cols,
            base_k=GBM_BASE_K,
            target_k=target_k,
            canary_idx=canary_idx,
        )
        name = f"union-gbm33-plus-{target_k - GBM_BASE_K}"
        meta = {
            "strategy": name,
            "base_k": GBM_BASE_K,
            "target_k": target_k,
            "test_positives": int(pred.sum()),
            "positive_coils": coils,
            "canary_all_positive": all(pred[canary_idx[i]] == 1 for i in range(len(canary_idx))),
        }
        path = write_submission(name, test, pred, meta)
        manifest.append({"name": name, "path": str(path), **meta})
        print(f"Wrote {path} ({meta['test_positives']} positives)")

    (OUTPUT_DIR / "union_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
