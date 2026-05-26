"""Random Forest with SMOTE inside CV and forum threshold t=0.31."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
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
from utils.threshold_tuning import apply_threshold, format_result, select_threshold_by_strategy

METHOD_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_SPLITS = 5
FEATURES = feature_names()
CLASS_WEIGHT = {0: 1, 1: 30}
FORUM_THRESHOLD = 0.31


def make_pipeline() -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=8,
                    class_weight=CLASS_WEIGHT,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--threshold-strategy",
        choices=("fixed_0.31", "target_rate", "auto"),
        default="fixed_0.31",
        help="Forum winners used t=0.31; also sweep 0.25–0.35 offline",
    )
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

    oof_proba = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_pr_aucs: list[float] = []

    for tr_idx, va_idx in skf.split(X_raw, y):
        pipe = make_pipeline()
        pipe.fit(X_raw[tr_idx], y[tr_idx])
        oof_proba[va_idx] = pipe.predict_proba(X_raw[va_idx])[:, 1]
        fold_pr_aucs.append(float(average_precision_score(y[va_idx], oof_proba[va_idx])))

    if args.threshold_strategy == "fixed_0.31":
        thresh = select_threshold_by_strategy(y, oof_proba, "fixed_0.31")
    elif args.threshold_strategy == "target_rate":
        thresh = select_threshold_by_strategy(y, oof_proba, "target_rate")
    else:
        thresh = select_threshold_by_strategy(y, oof_proba, "auto")

    print(format_result(thresh))
    oof_pred = apply_threshold(oof_proba, thresh.threshold)

    pd.DataFrame(
        {"CoilID": coil_ids, "y_true": y, "oof_proba": oof_proba, "oof_pred": oof_pred}
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    final_pipe = make_pipeline()
    final_pipe.fit(X_raw, y)
    joblib.dump(final_pipe, artifacts_dir / "pipeline.joblib")

    meta = {
        "method": "rf-smote",
        "threshold": thresh.threshold,
        "threshold_strategy": thresh.strategy,
        "class_weight": CLASS_WEIGHT,
        "forum_threshold": FORUM_THRESHOLD,
        "features": FEATURES,
        "smote_k_neighbors": 5,
    }
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    report = classification_report(y, oof_pred, output_dict=True, zero_division=0)
    metrics = {
        "method": "rf-smote",
        "oof_pr_auc": float(average_precision_score(y, oof_proba)),
        "oof_accuracy": thresh.accuracy,
        "oof_recall": thresh.recall,
        "best_threshold": thresh.threshold,
        "threshold_strategy": thresh.strategy,
        "majority_baseline_accuracy": majority_baseline,
        "fold_pr_auc": fold_pr_aucs,
        "classification_report": report,
        "train_rows": len(y),
        "prior_best_lb_score": 7.92453,
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"threshold": thresh.to_dict(), "class_weight": CLASS_WEIGHT})

    plot_pr_curve(y, oof_proba, plots_dir / "oof_pr_curve.png", "OOF RF+SMOTE PR")
    plot_threshold_sweep(
        y, oof_proba, plots_dir / "threshold_sweep.png",
        best_threshold=thresh.threshold, majority_baseline=majority_baseline,
    )
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"t={thresh.threshold:.3f}")
    plot_fold_scores({"PR-AUC": fold_pr_aucs}, plots_dir / "fold_pr_auc.png", "CV fold PR-AUC")

    copy_to_latest_summary(METHOD_DIR, run_dir)
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
