"""Predict test set from a saved lightgbm-tuned run."""

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
from utils.tabular_features import to_frame

METHOD_DIR = Path(__file__).resolve().parent


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
    pred = (proba >= meta["threshold"]).astype(int)

    submission = pd.DataFrame({"CoilID": test["CoilID"], "Y": pred})
    submission.to_csv(predictions_dir / "submission.csv", index=False)
    pd.DataFrame({"CoilID": test["CoilID"], "proba": proba, "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )

    meta_out = {"threshold": meta["threshold"], "test_positives": int((pred == 1).sum())}
    save_json(predictions_dir / "predict_meta.json", meta_out)

    if (run_dir / "metrics.json").is_file():
        metrics = load_json(run_dir / "metrics.json")
        metrics["test_positives"] = meta_out["test_positives"]
        save_json(run_dir / "metrics.json", metrics)

    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"Test positives: {meta_out['test_positives']} / {len(pred)}")


if __name__ == "__main__":
    main()
