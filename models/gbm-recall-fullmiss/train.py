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
METHOD_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from features import feature_names, to_frame  # noqa: E402

from utils.plotting import plot_confusion_matrix, plot_fold_scores, plot_pr_curve, plot_threshold_sweep
from utils.run_artifacts import (
    copy_to_latest_summary,
    create_run_dir,
    print_saved_artifacts,
    save_metrics,
    save_run_config,
    write_artifacts_manifest,
)
from utils.threshold_tuning import (
    apply_threshold,
    apply_top_k,
    format_result,
    select_recall_oriented_threshold,
    select_threshold_by_strategy,
)

RANDOM_STATE = 42
N_SPLITS = 5
N_SPLITS_SCALE = 10
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

XGB_PARAMS_SCALE = dict(
    n_estimators=2000, max_depth=5, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
)
LGBM_PARAMS_SCALE = dict(
    n_estimators=2000, max_depth=5, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
    objective="binary", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)
CATBOOST_PARAMS_SCALE = dict(
    iterations=2000, depth=5, learning_rate=0.02, subsample=0.8, random_seed=RANDOM_STATE,
    verbose=0, allow_writing_files=False, auto_class_weights="Balanced",
)


def get_model_params(scale: bool) -> tuple[dict, dict, dict]:
    if scale:
        return XGB_PARAMS_SCALE, LGBM_PARAMS_SCALE, CATBOOST_PARAMS_SCALE
    return XGB_PARAMS, LGBM_PARAMS, CATBOOST_PARAMS


def clone_model(name: str, scale_pos_weight: float, scale: bool = False):
    xgb_p, lgbm_p, cat_p = get_model_params(scale)
    if name == "xgb":
        return XGBClassifier(**xgb_p, scale_pos_weight=scale_pos_weight)
    if name == "lgbm":
        return LGBMClassifier(**lgbm_p, scale_pos_weight=scale_pos_weight)
    return CatBoostClassifier(**cat_p)


CANARY_COILS = (654, 806, 532, 958, 1187)


def canary_indices(coil_ids: np.ndarray) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--threshold-strategy",
        choices=("auto", "forum_fixed", "target_test_positives", "target_rate", "top_k_33"),
        default="auto",
        help="forum_fixed uses t=0.05; top_k_33 uses rank-based K=33",
    )
    parser.add_argument("--top-k", type=int, default=33, help="K for top_k_* strategies")
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Use scaled hyperparams (2000 trees, lr=0.02, depth=5, 10-fold CV)",
    )
    args = parser.parse_args()

    n_splits = N_SPLITS_SCALE if args.scale else N_SPLITS
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
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_pr_aucs: list[float] = []

    for tr_idx, va_idx in skf.split(X, y):
        for name in MODEL_NAMES:
            model = clone_model(name, scale_pos_weight, scale=args.scale)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            oof[name][va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        blend = sum(BLEND_WEIGHTS[n] * oof[n][va_idx] for n in MODEL_NAMES)
        fold_pr_aucs.append(float(average_precision_score(y[va_idx], blend)))

    oof_blend = sum(BLEND_WEIGHTS[n] * oof[n] for n in MODEL_NAMES)
    strategy = args.threshold_strategy
    if strategy == "top_k_33":
        strategy = f"top_k_{args.top_k}"
    if args.threshold_strategy == "auto":
        thresh = select_recall_oriented_threshold(y, oof_blend)
    else:
        thresh = select_threshold_by_strategy(y, oof_blend, strategy)
    print(format_result(thresh))
    if thresh.strategy.startswith("top_k_"):
        k = int(thresh.strategy.split("_")[-1])
        oof_pred, _ = apply_top_k(oof_blend, k, force_positive_idx=canary_indices(coil_ids))
    else:
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
        model = clone_model(name, scale_pos_weight, scale=args.scale)
        model.fit(X, y)
        final_models[name] = model

    final_models["xgb"].save_model(str(artifacts_dir / "xgb_model.json"))
    joblib.dump(final_models["lgbm"], artifacts_dir / "lgbm_model.joblib")
    final_models["catboost"].save_model(str(artifacts_dir / "catboost_model.cbm"))

    meta = {
        "method": "gbm-recall-fullmiss",
        "threshold": thresh.threshold,
        "threshold_strategy": thresh.strategy,
        "top_k": args.top_k if thresh.strategy.startswith("top_k_") else None,
        "blend_weights": BLEND_WEIGHTS,
        "features": FEATURES,
        "scaled": args.scale,
        "n_splits": n_splits,
    }
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    report = classification_report(y, oof_pred, output_dict=True, zero_division=0)
    metrics = {
        "method": "gbm-recall-fullmiss",
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
