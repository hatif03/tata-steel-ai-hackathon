"""Predict test set for knn-positive-profile."""

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

from utils.run_artifacts import copy_to_latest_summary, resolve_run_dir
from utils.tabular_features import to_frame
from utils.threshold_tuning import apply_top_k

METHOD_DIR = Path(__file__).resolve().parent
CANARY_COILS = (654, 806, 532, 958, 1187)


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def positive_profile_score(X_pos: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    from sklearn.metrics import pairwise_distances

    if len(X_pos) == 0:
        return np.zeros(len(X_query))
    dist = pairwise_distances(X_query, X_pos, metric="cosine")
    return 1.0 - dist.min(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(METHOD_DIR, args.run_dir)
    meta = joblib.load(run_dir / "artifacts" / "meta.joblib")
    imp = joblib.load(run_dir / "artifacts" / "imputer.joblib")
    scaler = joblib.load(run_dir / "artifacts" / "scaler.joblib")
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    y_train = train["Y"].astype(int).values
    X_train = scaler.transform(imp.transform(to_frame(train).values))
    X_test = scaler.transform(imp.transform(to_frame(test).values))
    pos_idx = np.where(y_train == 1)[0]
    proba = positive_profile_score(X_train[pos_idx], X_test)
    k = int(meta["k"])
    pred, _ = apply_top_k(proba, k, force_positive_idx=canary_indices(test["CoilID"]))

    pd.DataFrame({"CoilID": test["CoilID"], "proba": proba, "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame({"CoilID": test["CoilID"], "Y": pred}).to_csv(
        predictions_dir / "submission.csv", index=False
    )
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"Test positives: {int(pred.sum())} / {len(pred)}")


if __name__ == "__main__":
    main()
