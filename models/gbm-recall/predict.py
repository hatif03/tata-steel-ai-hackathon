"""Predict test set for gbm-recall ensemble."""

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
from utils.threshold_tuning import apply_threshold

METHOD_DIR = Path(__file__).resolve().parent
MODEL_NAMES = ("xgb", "lgbm", "catboost")
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
    weights = meta["blend_weights"]
    test = pd.read_csv(args.data_dir / "test.csv")
    X = to_frame(test)

    xgb = XGBClassifier()
    xgb.load_model(str(artifacts_dir / "xgb_model.json"))
    lgbm = joblib.load(artifacts_dir / "lgbm_model.joblib")
    cat = CatBoostClassifier()
    cat.load_model(str(artifacts_dir / "catboost_model.cbm"))

    proba = (
        weights["xgb"] * xgb.predict_proba(X)[:, 1]
        + weights["lgbm"] * lgbm.predict_proba(X)[:, 1]
        + weights["catboost"] * cat.predict_proba(X)[:, 1]
    )
    pred = apply_threshold(proba, meta["threshold"])

    pd.DataFrame({"CoilID": test["CoilID"], "Y": pred}).to_csv(
        predictions_dir / "submission.csv", index=False
    )
    pd.DataFrame({"CoilID": test["CoilID"], "proba": proba, "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )

    canary_mask = test["CoilID"].isin(CANARY_COILS)
    canary_preds = {
        int(k): int(v) for k, v in zip(
            test.loc[canary_mask, "CoilID"].astype(int),
            pred[canary_mask.to_numpy()],
        )
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
    print(f"Canary: {canary_preds}")


if __name__ == "__main__":
    main()
