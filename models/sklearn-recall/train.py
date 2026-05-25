"""RF + ExtraTrees + GradientBoosting ensemble with recall-first threshold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
from utils.threshold_tuning import apply_threshold, format_result, select_recall_oriented_threshold

METHOD_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_SPLITS = 5
FEATURES = feature_names()
MODEL_NAMES = ("rf", "et", "gbm")
CLASS_WEIGHT = {0: 1, 1: 30}
BLEND_WEIGHTS = {n: 1 / 3 for n in MODEL_NAMES}


def clone_model(name: str):
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
    args = parser.parse_args()

    run_dir = create_run_dir(METHOD_DIR, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    plots_dir = run_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    X_raw = to_frame(train).values
    y = train["Y"].astype(int).values
    coil_ids = train["CoilID"].values
    majority_baseline = float((y == 0).mean())

    oof = {n: np.zeros(len(y)) for n in MODEL_NAMES}
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_pr_aucs: list[float] = []

    for tr_idx, va_idx in skf.split(X_raw, y):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_raw[tr_idx])
        X_va = imp.transform(X_raw[va_idx])
        for name in MODEL_NAMES:
            model = clone_model(name)
            model.fit(X_tr, y[tr_idx])
            oof[name][va_idx] = model.predict_proba(X_va)[:, 1]
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
            "oof_rf": oof["rf"],
            "oof_et": oof["et"],
            "oof_gbm": oof["gbm"],
            "oof_blend": oof_blend,
            "oof_pred": oof_pred,
        }
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    imp_full = SimpleImputer(strategy="median")
    X_full = imp_full.fit_transform(X_raw)
    joblib.dump(imp_full, artifacts_dir / "imputer.joblib")

    for name in MODEL_NAMES:
        model = clone_model(name)
        model.fit(X_full, y)
        joblib.dump(model, artifacts_dir / f"{name}_model.joblib")

    meta = {
        "method": "sklearn-recall",
        "threshold": thresh.threshold,
        "threshold_strategy": thresh.strategy,
        "blend_weights": BLEND_WEIGHTS,
        "class_weight": CLASS_WEIGHT,
        "features": FEATURES,
    }
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    report = classification_report(y, oof_pred, output_dict=True, zero_division=0)
    metrics = {
        "method": "sklearn-recall",
        "oof_pr_auc": float(average_precision_score(y, oof_blend)),
        "oof_accuracy": thresh.accuracy,
        "oof_recall": thresh.recall,
        "best_threshold": thresh.threshold,
        "threshold_strategy": thresh.strategy,
        "majority_baseline_accuracy": majority_baseline,
        "fold_pr_auc": fold_pr_aucs,
        "classification_report": report,
        "train_rows": len(y),
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"class_weight": CLASS_WEIGHT, "threshold": thresh.to_dict()})

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
