"""50/50 blend of sklearn-recall trio and LightGBM with recall-oriented threshold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import StratifiedKFold

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
from utils.threshold_tuning import (
    apply_threshold,
    format_result,
    select_threshold_by_strategy,
)

METHOD_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_SPLITS = 5
FEATURES = feature_names()
SKLEARN_NAMES = ("rf", "et", "gbm")
CLASS_WEIGHT = {0: 1, 1: 30}
SKLEARN_BLEND = {n: 1 / 3 for n in SKLEARN_NAMES}
BLEND_WEIGHTS = {"sklearn": 0.5, "lgbm": 0.5}

LGBM_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)


def clone_sklearn(name: str):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500, max_depth=8, class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    if name == "et":
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=8, class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    return GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--threshold-strategy",
        choices=("target_rate", "target_test_positives", "auto"),
        default="target_rate",
    )
    args = parser.parse_args()

    run_dir = create_run_dir(METHOD_DIR, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    plots_dir = run_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    X_df = to_frame(train)
    X_raw = X_df.values
    y = train["Y"].astype(int).values
    coil_ids = train["CoilID"].values
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    majority_baseline = float((y == 0).mean())
    lgbm_params = {**LGBM_PARAMS, "scale_pos_weight": scale_pos_weight}

    oof_sklearn = {n: np.zeros(len(y)) for n in SKLEARN_NAMES}
    oof_lgbm = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_pr_aucs: list[float] = []

    for tr_idx, va_idx in skf.split(X_raw, y):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_raw[tr_idx])
        X_va = imp.transform(X_raw[va_idx])
        X_tr_lgb = X_df.iloc[tr_idx]
        X_va_lgb = X_df.iloc[va_idx]

        for name in SKLEARN_NAMES:
            model = clone_sklearn(name)
            model.fit(X_tr, y[tr_idx])
            oof_sklearn[name][va_idx] = model.predict_proba(X_va)[:, 1]

        lgbm = LGBMClassifier(**lgbm_params)
        lgbm.fit(X_tr_lgb, y[tr_idx])
        oof_lgbm[va_idx] = lgbm.predict_proba(X_va_lgb)[:, 1]

        sk_blend = sum(SKLEARN_BLEND[n] * oof_sklearn[n][va_idx] for n in SKLEARN_NAMES)
        fold_blend = BLEND_WEIGHTS["sklearn"] * sk_blend + BLEND_WEIGHTS["lgbm"] * oof_lgbm[va_idx]
        fold_pr_aucs.append(float(average_precision_score(y[va_idx], fold_blend)))

    oof_sklearn_blend = sum(SKLEARN_BLEND[n] * oof_sklearn[n] for n in SKLEARN_NAMES)
    oof_blend = BLEND_WEIGHTS["sklearn"] * oof_sklearn_blend + BLEND_WEIGHTS["lgbm"] * oof_lgbm

    if args.threshold_strategy == "target_rate":
        thresh = select_threshold_by_strategy(y, oof_blend, "target_rate")
    else:
        thresh = select_threshold_by_strategy(y, oof_blend, "auto")

    print(format_result(thresh))
    oof_pred = apply_threshold(oof_blend, thresh.threshold)

    pd.DataFrame(
        {
            "CoilID": coil_ids,
            "y_true": y,
            "oof_sklearn": oof_sklearn_blend,
            "oof_lgbm": oof_lgbm,
            "oof_blend": oof_blend,
            "oof_pred": oof_pred,
        }
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    imp_full = SimpleImputer(strategy="median")
    X_full = imp_full.fit_transform(X_raw)
    joblib.dump(imp_full, artifacts_dir / "imputer.joblib")
    for name in SKLEARN_NAMES:
        model = clone_sklearn(name)
        model.fit(X_full, y)
        joblib.dump(model, artifacts_dir / f"{name}_model.joblib")

    final_lgbm = LGBMClassifier(**lgbm_params)
    final_lgbm.fit(X_df, y)
    joblib.dump(final_lgbm, artifacts_dir / "lgbm_model.joblib")

    meta = {
        "method": "recall-blend",
        "threshold": thresh.threshold,
        "threshold_strategy": thresh.strategy,
        "blend_weights": BLEND_WEIGHTS,
        "sklearn_blend_weights": SKLEARN_BLEND,
        "class_weight": CLASS_WEIGHT,
        "features": FEATURES,
        "lgbm_params": lgbm_params,
    }
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    report = classification_report(y, oof_pred, output_dict=True, zero_division=0)
    metrics = {
        "method": "recall-blend",
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
        "prior_best_lb_score": 7.92453,
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"blend_weights": BLEND_WEIGHTS, "threshold": thresh.to_dict()})

    plot_pr_curve(y, oof_blend, plots_dir / "oof_pr_curve.png", "OOF recall-blend PR")
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
