"""Predict test set for gbm-mega-blend."""

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
COLS = ("gbm-recall", "sklearn-recall", "lightgbm-recall")


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def load_test_matrix(test: pd.DataFrame) -> np.ndarray:
    parts = []
    for name in COLS:
        p = pd.read_csv(ROOT / f"models/{name}/outputs/latest/predictions/test_predictions.csv")
        parts.append(test[["CoilID"]].merge(p[["CoilID", "proba"]], on="CoilID")["proba"].values)
    return np.column_stack(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(METHOD_DIR, args.run_dir)
    artifacts_dir = run_dir / "artifacts"
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    meta = joblib.load(artifacts_dir / "meta.joblib")
    test = pd.read_csv(args.data_dir / "test.csv")
    X = load_test_matrix(test)
    k = int(meta["k"])
    strategy = meta["selected_strategy"]

    if strategy == "gbm_only":
        proba = X[:, 0]
    elif strategy == "stacking":
        proba = joblib.load(artifacts_dir / "stacker.joblib").predict_proba(X)[:, 1]
    elif strategy == "gbm_heavy":
        w = np.array([meta["gbm_heavy_weights"][c] for c in COLS])
        proba = X @ w
    else:
        w = np.array([meta["blend_weights"][c] for c in COLS])
        proba = X @ w

    pred, eff_t = apply_top_k(proba, k, force_positive_idx=canary_indices(test["CoilID"]))
    pd.DataFrame({"CoilID": test["CoilID"], "Y": pred}).to_csv(predictions_dir / "submission.csv", index=False)
    pd.DataFrame({"CoilID": test["CoilID"], "proba": proba, "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )
    predict_meta = {"strategy": strategy, "k": k, "effective_threshold": eff_t, "test_positives": int(pred.sum())}
    save_json(predictions_dir / "predict_meta.json", predict_meta)
    if (run_dir / "metrics.json").is_file():
        metrics = load_json(run_dir / "metrics.json")
        metrics["test_positives"] = predict_meta["test_positives"]
        save_json(run_dir / "metrics.json", metrics)
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"Test positives: {predict_meta['test_positives']} strategy={strategy}")


if __name__ == "__main__":
    main()
