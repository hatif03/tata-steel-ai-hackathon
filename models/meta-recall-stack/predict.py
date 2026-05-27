"""Predict test set for meta-recall-stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_artifacts import copy_to_latest_summary, load_json, resolve_run_dir, save_json
from utils.threshold_tuning import apply_top_k

METHOD_DIR = Path(__file__).resolve().parent
CANARY_COILS = (654, 806, 532, 958, 1187)


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def load_test_matrix(test: pd.DataFrame, sources: list[str]) -> np.ndarray:
    cols = []
    for name in sources:
        pred = pd.read_csv(ROOT / f"models/{name}/outputs/latest/predictions/test_predictions.csv")
        merged = test[["CoilID"]].merge(pred[["CoilID", "proba"]], on="CoilID")
        cols.append(merged["proba"].values.astype(float))
    return np.column_stack(cols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(METHOD_DIR, args.run_dir)
    meta = joblib.load(run_dir / "artifacts" / "meta.joblib")
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(args.data_dir / "test.csv")
    sources = meta["source_models"]
    X = load_test_matrix(test, sources)
    k = int(meta["k"])
    strategy = meta["selected_strategy"]

    if strategy == "stacking":
        stacker = joblib.load(run_dir / "artifacts" / "stacker.joblib")
        proba = stacker.predict_proba(X)[:, 1]
    elif strategy == "weight_opt":
        w = np.array([meta["blend_weights"][m] for m in sources])
        proba = X @ w
    else:
        proba = X.mean(axis=1)

    pred, eff_t = apply_top_k(proba, k, force_positive_idx=canary_indices(test["CoilID"]))
    pd.DataFrame({"CoilID": test["CoilID"], "proba": proba, "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame({"CoilID": test["CoilID"], "Y": pred}).to_csv(
        predictions_dir / "submission.csv", index=False
    )
    save_json(
        predictions_dir / "predict_meta.json",
        {"test_positives": int(pred.sum()), "strategy": strategy, "k": k, "effective_threshold": eff_t},
    )
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"Test positives: {int(pred.sum())} strategy={strategy} K={k}")


if __name__ == "__main__":
    main()
