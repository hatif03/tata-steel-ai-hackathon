"""Equal-weight XGB + LightGBM + CatBoost with recall-first threshold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plotting import plot_confusion_matrix, plot_fold_scores, plot_pr_curve, plot_threshold_sweep
from utils.run_artifacts import (
    copy_to_latest_summary,
    create_run_dir,
    print_saved_artifacts,
    save_metrics,
    save_run_config,
    write_artifacts_manifest,
)
from utils.tabular_features import feature_names, to_frame
from utils.threshold_tuning import apply_threshold, format_result, select_recall_oriented_threshold

METHOD_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_SPLITS = 5
FEATURES = feature_names()
MODEL_NAMES = ("xgb", "lgbm", "catboost")
BLEND_WEIGHTS = {"xgb": 1 / 3, "lgbm": 1 / 3, "catboost": 1 / 3}

XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
)
LGBM_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    objective="binary", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)
CATBOOST_PARAMS = dict(
    iterations=500, depth=4, learning_rate=0.05, subsample=0.8, random_seed=RANDOM_STATE,
    verbose=0, allow_writing_files=False, auto_class_weights="Balanced",
)


def clone_model(name: str, scale_pos_weight: float):
    if name == "xgb":
        return XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    if name == "lgbm":
        return LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=scale_pos_weight)
    return CatBoostClassifier(**CATBOOST_PARAMS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    run_dir = create_run_dir(METHOD_DIR, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    plots_dir = run_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    X = to_frame(train)
    y = train["Y"].astype(int).values
    coil_ids = train["CoilID"].values
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    majority_baseline = float((y == 0).mean())

    oof = {n: np.zeros(len(y)) for n in MODEL_NAMES}
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_pr_aucs: list[float] = []

    for tr_idx, va_idx in skf.split(X, y):
        for name in MODEL_NAMES:
            model = clone_model(name, scale_pos_weight)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            oof[name][va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        blend = sum(BLEND_WEIGHTS[n] * oof[n][va_idx] for n in MODEL_NAMES)
        fold_pr_aucs.append(float(average_precision_score(y[va_idx], blend)))

    oof_blend = sum(BLEND_WEIGHTS[n] * oof[n] for n in MODEL_NAMES)
    thresh = select_recall_oriented_threshold(y, oof_blend)
    print(format_result(thresh))
    oof_pred = apply_threshold(oof_blend, thresh.threshold)

    pd.DataFrame(
        {
            "CoilID": coil_ids,
            "y_true": y,
            "oof_xgb": oof["xgb"],
            "oof_lgbm": oof["lgbm"],
            "oof_catboost": oof["catboost"],
            "oof_blend": oof_blend,
            "oof_pred": oof_pred,
        }
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    final_models = {}
    for name in MODEL_NAMES:
        model = clone_model(name, scale_pos_weight)
        model.fit(X, y)
        final_models[name] = model

    final_models["xgb"].save_model(str(artifacts_dir / "xgb_model.json"))
    joblib.dump(final_models["lgbm"], artifacts_dir / "lgbm_model.joblib")
    final_models["catboost"].save_model(str(artifacts_dir / "catboost_model.cbm"))

    meta = {
        "method": "gbm-recall",
        "threshold": thresh.threshold,
        "threshold_strategy": thresh.strategy,
        "blend_weights": BLEND_WEIGHTS,
        "features": FEATURES,
    }
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    report = classification_report(y, oof_pred, output_dict=True, zero_division=0)
    metrics = {
        "method": "gbm-recall",
        "oof_pr_auc": float(average_precision_score(y, oof_blend)),
        "oof_accuracy": thresh.accuracy,
        "oof_recall": thresh.recall,
        "best_threshold": thresh.threshold,
        "threshold_strategy": thresh.strategy,
        "majority_baseline_accuracy": majority_baseline,
        "fold_pr_auc": fold_pr_aucs,
        "classification_report": report,
        "train_rows": len(y),
        "blend_weights": BLEND_WEIGHTS,
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"blend_weights": BLEND_WEIGHTS, "threshold": thresh.to_dict()})

    plot_pr_curve(y, oof_blend, plots_dir / "oof_pr_curve.png", "OOF blend PR")
    plot_threshold_sweep(
        y, oof_blend, plots_dir / "threshold_sweep.png",
        best_threshold=thresh.threshold, majority_baseline=majority_baseline,
    )
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"t={thresh.threshold:.3f}")
    plot_fold_scores({"PR-AUC": fold_pr_aucs}, plots_dir / "fold_pr_auc.png", "CV fold PR-AUC")

    copy_to_latest_summary(METHOD_DIR, run_dir)
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
