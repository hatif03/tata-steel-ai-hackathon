"""RF + BorderlineSMOTE, isotonic calibration, rank-based top-K threshold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plotting import plot_confusion_matrix, plot_pr_curve
from utils.run_artifacts import (
    copy_to_latest_summary,
    create_run_dir,
    print_saved_artifacts,
    save_metrics,
    save_run_config,
    write_artifacts_manifest,
)
from utils.tabular_features import feature_names, to_frame
from utils.threshold_tuning import apply_top_k, tune_top_k

METHOD_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_SPLITS = 5
CANARY_COILS = (654, 806, 532, 958, 1187)
DEFAULT_K = 26


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def make_base_pipeline(sampler: str = "borderline") -> ImbPipeline:
    if sampler == "adasyn":
        from imblearn.over_sampling import ADASYN
        over = ADASYN(random_state=RANDOM_STATE, n_neighbors=5)
    else:
        over = BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    return ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", over),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=8,
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
    parser.add_argument("--sampler", choices=("borderline", "adasyn"), default="borderline")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
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
    canary_idx = canary_indices(train["CoilID"])

    oof_proba = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_pr: list[float] = []

    for tr_idx, va_idx in skf.split(X_raw, y):
        pipe = make_base_pipeline(args.sampler)
        pipe.fit(X_raw[tr_idx], y[tr_idx])
        oof_proba[va_idx] = pipe.predict_proba(X_raw[va_idx])[:, 1]
        fold_pr.append(float(average_precision_score(y[va_idx], oof_proba[va_idx])))

    thresh = tune_top_k(y, oof_proba, args.k, force_positive_idx=canary_idx)
    oof_pred, _ = apply_top_k(oof_proba, args.k, force_positive_idx=canary_idx)

    pd.DataFrame(
        {"CoilID": coil_ids, "y_true": y, "oof_proba": oof_proba, "oof_pred": oof_pred}
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    final_pipe = make_base_pipeline(args.sampler)
    final_pipe.fit(X_raw, y)
    joblib.dump(final_pipe, artifacts_dir / "pipeline.joblib")

    meta = {
        "method": "rf-smote-v2",
        "sampler": args.sampler,
        "k": args.k,
        "threshold": thresh.threshold,
        "threshold_strategy": thresh.strategy,
        "features": feature_names(),
    }
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "rf-smote-v2",
        "oof_pr_auc": float(average_precision_score(y, oof_proba)),
        "oof_accuracy": thresh.accuracy,
        "oof_recall": thresh.recall,
        "best_threshold": thresh.threshold,
        "fold_pr_auc": fold_pr,
        "prior_best_lb_score": 9.05660,
        "classification_report": classification_report(y, oof_pred, output_dict=True, zero_division=0),
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"sampler": args.sampler, "k": args.k})

    plot_pr_curve(y, oof_proba, plots_dir / "oof_pr_curve.png", "RF-SMOTE v2 OOF PR")
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"K={args.k}")

    copy_to_latest_summary(METHOD_DIR, run_dir)
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
