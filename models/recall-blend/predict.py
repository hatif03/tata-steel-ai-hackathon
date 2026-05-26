"""Predict test set for recall-blend (50/50 sklearn + LightGBM)."""

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
from utils.tabular_features import to_frame
from utils.threshold_tuning import apply_threshold

METHOD_DIR = Path(__file__).resolve().parent
SKLEARN_NAMES = ("rf", "et", "gbm")
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
    sk_weights = meta["sklearn_blend_weights"]
    blend_weights = meta["blend_weights"]
    test = pd.read_csv(args.data_dir / "test.csv")
    X_raw = joblib.load(artifacts_dir / "imputer.joblib").transform(to_frame(test).values)
    X_lgb = to_frame(test)

    sk_proba = np.zeros(len(test))
    for name in SKLEARN_NAMES:
        model = joblib.load(artifacts_dir / f"{name}_model.joblib")
        sk_proba += sk_weights[name] * model.predict_proba(X_raw)[:, 1]

    lgbm = joblib.load(artifacts_dir / "lgbm_model.joblib")
    lgb_proba = lgbm.predict_proba(X_lgb)[:, 1]
    proba = blend_weights["sklearn"] * sk_proba + blend_weights["lgbm"] * lgb_proba

    pred = apply_threshold(proba, meta["threshold"])
    pd.DataFrame({"CoilID": test["CoilID"], "Y": pred}).to_csv(
        predictions_dir / "submission.csv", index=False
    )
    pd.DataFrame({"CoilID": test["CoilID"], "proba": proba, "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )

    canary_mask = test["CoilID"].isin(CANARY_COILS)
    canary_preds = {
        int(c): int(p)
        for c, p in zip(test.loc[canary_mask, "CoilID"], pred[canary_mask.to_numpy()])
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
