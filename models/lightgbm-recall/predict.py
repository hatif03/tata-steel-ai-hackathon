"""Predict test set for lightgbm-recall."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHOD_DIR = Path(__file__).resolve().parent
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from features import to_frame  # noqa: E402

from utils.run_artifacts import copy_to_latest_summary, load_json, resolve_run_dir, save_json
from utils.threshold_tuning import apply_threshold

CANARY_COILS = (654, 806, 532, 958, 1187)


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
    model = joblib.load(artifacts_dir / "model.joblib")
    proba = model.predict_proba(to_frame(test))[:, 1]
    pred = apply_threshold(proba, meta["threshold"])

    pd.DataFrame({"CoilID": test["CoilID"], "Y": pred}).to_csv(
        predictions_dir / "submission.csv", index=False
    )
    pd.DataFrame({"CoilID": test["CoilID"], "proba": proba, "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )

    canary = test[test["CoilID"].isin(CANARY_COILS)].copy()
    canary["proba"] = model.predict_proba(to_frame(canary))[:, 1]
    canary["Y"] = apply_threshold(canary["proba"].values, meta["threshold"])

    canary_preds = {
        int(c): int(p)
        for c, p in zip(canary["CoilID"].astype(int), canary["Y"].astype(int))
    }
    predict_meta = {
        "threshold": meta["threshold"],
        "threshold_strategy": meta.get("threshold_strategy"),
        "test_positives": int((pred == 1).sum()),
        "canary_predictions": canary_preds,
    }
    save_json(predictions_dir / "predict_meta.json", predict_meta)

    if (run_dir / "metrics.json").is_file():
        metrics = load_json(run_dir / "metrics.json")
        metrics["test_positives"] = predict_meta["test_positives"]
        save_json(run_dir / "metrics.json", metrics)

    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"Test positives: {predict_meta['test_positives']} / {len(pred)}")
    print(f"Canary: {predict_meta['canary_predictions']}")


if __name__ == "__main__":
    main()
