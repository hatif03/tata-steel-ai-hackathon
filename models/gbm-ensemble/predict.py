"""Generate submission CSV from a saved GBM ensemble run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_artifacts import copy_to_latest_summary, load_json, resolve_run_dir, save_json
from utils.tabular_features import to_frame

METHOD_DIR = Path(__file__).resolve().parent
MODEL_NAMES = ("xgb", "lgbm", "catboost")


def load_models(artifacts_dir: Path) -> dict[str, object]:
    xgb = XGBClassifier()
    xgb.load_model(str(artifacts_dir / "xgb_model.json"))
    lgbm = joblib.load(artifacts_dir / "lgbm_model.joblib")
    cb_path = artifacts_dir / "catboost_model.cbm"
    if cb_path.is_file():
        catboost = CatBoostClassifier()
        catboost.load_model(str(cb_path))
    else:
        catboost = joblib.load(artifacts_dir / "catboost_model.joblib")
    return {"xgb": xgb, "lgbm": lgbm, "catboost": catboost}


def blend_predict(models: dict[str, object], X: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    proba = np.zeros(len(X))
    for name in MODEL_NAMES:
        proba += weights[name] * models[name].predict_proba(X)[:, 1]
    return proba


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict test set from a saved ensemble run")
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
    X_test = to_frame(test)
    models = load_models(artifacts_dir)
    weights = meta["blend_weights"]
    threshold = meta["threshold"]

    proba = blend_predict(models, X_test, weights)
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
        "blend_weights": weights,
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
    print(f"Blend weights: {weights}")
    print(f"Test positives: {predict_meta['test_positives']} / {predict_meta['test_rows']}")


if __name__ == "__main__":
    main()
