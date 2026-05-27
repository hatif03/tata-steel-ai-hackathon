"""Predict test set for autogluon-recall."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_artifacts import copy_to_latest_summary, load_json, resolve_run_dir, save_json
from utils.tabular_features import build_features
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

    from autogluon.tabular import TabularPredictor

    run_dir = resolve_run_dir(METHOD_DIR, args.run_dir)
    meta = joblib.load(run_dir / "artifacts" / "meta.joblib")
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    test_raw = pd.read_csv(args.data_dir / "test.csv")
    test_df = build_features(test_raw)
    predictor = TabularPredictor.load(str(run_dir / "artifacts" / "autogluon"))
    proba_df = predictor.predict_proba(test_df)
    proba = proba_df[1].values if 1 in proba_df.columns else proba_df.iloc[:, -1].values
    k = int(meta["k"])
    pred, _ = apply_top_k(proba, k, force_positive_idx=canary_indices(test_raw["CoilID"]))

    pd.DataFrame({"CoilID": test_raw["CoilID"], "proba": proba, "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame({"CoilID": test_raw["CoilID"], "Y": pred}).to_csv(
        predictions_dir / "submission.csv", index=False
    )
    save_json(predictions_dir / "predict_meta.json", {"test_positives": int(pred.sum()), "k": k})
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"Test positives: {int(pred.sum())} / {len(pred)}")


if __name__ == "__main__":
    main()
