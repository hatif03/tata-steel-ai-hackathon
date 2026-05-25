"""Train LightGBM with recall-oriented threshold tuning (Phase 0 evolution)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHOD_DIR = Path(__file__).resolve().parent
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
from utils.threshold_tuning import apply_threshold, format_result, select_recall_oriented_threshold

RANDOM_STATE = 42
N_SPLITS = 5
FEATURES = feature_names()

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
    params = {**LGBM_PARAMS, "scale_pos_weight": scale_pos_weight}

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y))
    fold_pr_aucs: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        model = LGBMClassifier(**params)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof_proba[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        fold_pr_aucs.append(float(average_precision_score(y[va_idx], oof_proba[va_idx])))
        print(f"fold {fold} PR-AUC: {fold_pr_aucs[-1]:.4f}")

    thresh = select_recall_oriented_threshold(y, oof_proba)
    print(format_result(thresh))

    oof_pred = apply_threshold(oof_proba, thresh.threshold)
    oof_pr_auc = float(average_precision_score(y, oof_proba))
    report = classification_report(y, oof_pred, output_dict=True, zero_division=0)

    pd.DataFrame(
        {"CoilID": coil_ids, "y_true": y, "oof_proba": oof_proba, "oof_pred": oof_pred}
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    final_model = LGBMClassifier(**params)
    final_model.fit(X, y)
    final_model.booster_.save_model(str(artifacts_dir / "lgbm_model.txt"))
    joblib.dump(final_model, artifacts_dir / "model.joblib")

    meta = {
        "method": "lightgbm-recall",
        "threshold": thresh.threshold,
        "threshold_strategy": thresh.strategy,
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": RANDOM_STATE,
        "features": FEATURES,
    }
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "lightgbm-recall",
        "oof_pr_auc": oof_pr_auc,
        "oof_accuracy": thresh.accuracy,
        "oof_recall": thresh.recall,
        "oof_precision": thresh.precision,
        "best_threshold": thresh.threshold,
        "threshold_strategy": thresh.strategy,
        "majority_baseline_accuracy": majority_baseline,
        "fold_pr_auc": fold_pr_aucs,
        "classification_report": report,
        "train_rows": len(y),
        "positive_rate": float(y.mean()),
        "oof_positives_predicted": thresh.n_predicted_positive,
        "prior_best_lb_score": 1.88679,
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"hyperparameters": params, "threshold_result": thresh.to_dict()})

    plot_pr_curve(y, oof_proba, plots_dir / "oof_pr_curve.png", "OOF PR")
    plot_threshold_sweep(
        y, oof_proba, plots_dir / "threshold_sweep.png",
        best_threshold=thresh.threshold, majority_baseline=majority_baseline,
    )
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"t={thresh.threshold:.3f}")
    plot_fold_scores({"PR-AUC": fold_pr_aucs}, plots_dir / "fold_pr_auc.png", "CV fold PR-AUC")

    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"OOF accuracy @ t={thresh.threshold:.4f}: {thresh.accuracy:.4f}")
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
