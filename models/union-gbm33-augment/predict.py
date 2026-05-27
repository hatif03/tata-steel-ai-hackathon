"""Predict using saved union-gbm33-augment meta (regenerates from source probas)."""

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
from utils.union_augment import build_augmented, canary_indices, merge_test_probas

METHOD_DIR = Path(__file__).resolve().parent
CANARY_COILS = (654, 806, 532, 958, 1187)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(METHOD_DIR, args.run_dir)
    meta = joblib.load(run_dir / "artifacts" / "meta.joblib")
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    secondary = tuple(meta.get("secondary_methods", []))
    test, loaded = merge_test_probas(ROOT, secondary_methods=secondary)
    canary_idx = canary_indices(test["CoilID"])
    pred, coils, added = build_augmented(
        test,
        "gbm-recall",
        secondary,
        loaded,
        base_k=meta["base_k"],
        target_k=meta["target_k"],
        secondary_k=meta["secondary_k"],
        canary_idx=canary_idx,
        ranking=meta.get("ranking", "max"),
    )

    test_df = pd.read_csv(args.data_dir / "test.csv")
    pd.DataFrame({"CoilID": test_df["CoilID"], "proba": test["gbm-recall"], "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame({"CoilID": test_df["CoilID"], "Y": pred}).to_csv(
        predictions_dir / "submission.csv", index=False
    )

    predict_meta = {
        "test_positives": int(pred.sum()),
        "positive_coils": coils,
        "added_exclusives": added,
    }
    save_json(predictions_dir / "predict_meta.json", predict_meta)

    if (run_dir / "metrics.json").is_file():
        metrics = load_json(run_dir / "metrics.json")
        metrics["test_positives"] = predict_meta["test_positives"]
        save_json(run_dir / "metrics.json", metrics)

    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"Test positives: {predict_meta['test_positives']} / {len(pred)}")


if __name__ == "__main__":
    main()
