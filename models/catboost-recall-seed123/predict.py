"""Predict test set for catboost-recall-seed123."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_artifacts import copy_to_latest_summary, resolve_run_dir, save_json
from utils.tabular_features import to_frame
from utils.threshold_tuning import apply_top_k

METHOD_DIR = Path(__file__).resolve().parent
CANARY_COILS = (654, 806, 532, 958, 1187)


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


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
    X = to_frame(test)
    model = CatBoostClassifier()
    model.load_model(str(run_dir / "artifacts" / "catboost_model.cbm"))
    proba = model.predict_proba(X)[:, 1]
    k = int(meta["k"])
    pred, _ = apply_top_k(proba, k, force_positive_idx=canary_indices(test["CoilID"]))

    pd.DataFrame({"CoilID": test["CoilID"], "proba": proba, "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame({"CoilID": test["CoilID"], "Y": pred}).to_csv(
        predictions_dir / "submission.csv", index=False
    )
    save_json(predictions_dir / "predict_meta.json", {"test_positives": int(pred.sum()), "k": k})
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"Test positives: {int(pred.sum())} / {len(pred)}")


if __name__ == "__main__":
    main()
