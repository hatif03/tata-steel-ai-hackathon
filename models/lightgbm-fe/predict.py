"""Generate submission CSV from a saved lightgbm-fe run."""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict test set from a saved run")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--use-latest-summary", action="store_true")
    args = parser.parse_args()

    if args.use_latest_summary:
        run_dir = METHOD_DIR / "outputs" / "latest"
        if not run_dir.is_dir():
            raise FileNotFoundError("No outputs/latest/ — run train.py first")
    else:
        run_dir = resolve_run_dir(METHOD_DIR, args.run_dir)

    artifacts_dir = run_dir / "artifacts"
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    meta_path = artifacts_dir / "meta.joblib"
    meta = joblib.load(meta_path) if meta_path.is_file() else load_json(artifacts_dir / "meta.json")

    test = pd.read_csv(args.data_dir / "test.csv")
    model = joblib.load(artifacts_dir / "model.joblib")
    proba = model.predict_proba(to_frame(test))[:, 1]
    threshold = meta["threshold"]
    pred = (proba >= threshold).astype(int)

    submission = pd.DataFrame({"CoilID": test["CoilID"], "Y": pred})
    submission_path = predictions_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    pd.DataFrame({"CoilID": test["CoilID"], "proba": proba, "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )

    predict_meta = {
        "submission_path": str(submission_path.resolve()),
        "threshold": threshold,
        "test_rows": len(submission),
        "test_positives": int((pred == 1).sum()),
    }
    save_json(predictions_dir / "predict_meta.json", predict_meta)

    if (run_dir / "metrics.json").is_file():
        metrics = load_json(run_dir / "metrics.json")
        metrics["test_positives"] = predict_meta["test_positives"]
        save_json(run_dir / "metrics.json", metrics)

    copy_to_latest_summary(METHOD_DIR, run_dir)

    print(f"Wrote {submission_path}")
    print(f"Test positives: {predict_meta['test_positives']} / {predict_meta['test_rows']}")


if __name__ == "__main__":
    main()
