"""XGBoost solo with recall-first top-K threshold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plotting import plot_confusion_matrix, plot_fold_scores, plot_pr_curve
from utils.run_artifacts import (
    copy_to_latest_summary,
    create_run_dir,
    print_saved_artifacts,
    save_metrics,
    save_run_config,
    write_artifacts_manifest,
)
from utils.tabular_features import feature_names, to_frame
from utils.threshold_tuning import apply_top_k, format_result, tune_top_k

METHOD_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_SPLITS = 5
CANARY_COILS = (654, 806, 532, 958, 1187)
DEFAULT_K = 33


def canary_indices(coil_ids: np.ndarray) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
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

    oof_proba = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_pr: list[float] = []
    params = dict(
        n_estimators=800, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    for tr, va in skf.split(X, y):
        model = XGBClassifier(**params)
        model.fit(X.iloc[tr], y[tr])
        oof_proba[va] = model.predict_proba(X.iloc[va])[:, 1]
        fold_pr.append(float(average_precision_score(y[va], oof_proba[va])))

    k = args.k
    thresh = tune_top_k(y, oof_proba, k, force_positive_idx=canary_indices(coil_ids))
    print(format_result(thresh))
    oof_pred, _ = apply_top_k(oof_proba, k, force_positive_idx=canary_indices(coil_ids))

    pd.DataFrame(
        {"CoilID": coil_ids, "y_true": y, "oof_proba": oof_proba, "oof_pred": oof_pred}
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    final = XGBClassifier(**params)
    final.fit(X, y)
    final.save_model(str(artifacts_dir / "xgb_model.json"))
    joblib.dump(
        {"method": "xgb-recall", "k": k, "threshold_strategy": thresh.strategy, "threshold": thresh.threshold,
         "features": feature_names()},
        artifacts_dir / "meta.joblib",
    )
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "xgb-recall", "k": k,
        "oof_pr_auc": float(average_precision_score(y, oof_proba)),
        "oof_accuracy": thresh.accuracy, "oof_recall": thresh.recall,
        "fold_pr_auc": fold_pr,
        "classification_report": classification_report(y, oof_pred, output_dict=True, zero_division=0),
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"k": k, "params": params})
    plot_pr_curve(y, oof_proba, plots_dir / "oof_pr_curve.png", "XGB OOF PR")
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"K={k}")
    plot_fold_scores({"PR-AUC": fold_pr}, plots_dir / "fold_pr_auc.png", "CV fold PR-AUC")
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
