"""Seed-averaged LightGBM on the proven lightgbm-cv feature pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, average_precision_score, classification_report
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

METHOD_DIR = Path(__file__).resolve().parent
N_SPLITS = 5
FEATURES = feature_names()
SEEDS = (42, 123, 456)

LGBM_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    n_jobs=-1,
    verbose=-1,
)


def tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        acc = accuracy_score(y_true, (proba >= t).astype(int))
        if acc > best_acc:
            best_t, best_acc = float(t), float(acc)
    return best_t, best_acc


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

    oof_blend = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    for seed in SEEDS:
        oof_seed = np.zeros(len(y))
        for tr_idx, va_idx in skf.split(X, y):
            model = LGBMClassifier(**LGBM_PARAMS, random_state=seed, scale_pos_weight=scale_pos_weight)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            oof_seed[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof_blend += oof_seed / len(SEEDS)
        print(f"seed {seed} OOF PR-AUC: {average_precision_score(y, oof_seed):.4f}")

    best_t, best_acc = tune_threshold(y, oof_blend)
    oof_pr_auc = float(average_precision_score(y, oof_blend))
    oof_pred = (oof_blend >= best_t).astype(int)
    report = classification_report(y, oof_pred, output_dict=True, zero_division=0)

    pd.DataFrame(
        {"CoilID": coil_ids, "y_true": y, "oof_proba": oof_blend, "oof_pred": oof_pred}
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    final_models = []
    for seed in SEEDS:
        model = LGBMClassifier(**LGBM_PARAMS, random_state=seed, scale_pos_weight=scale_pos_weight)
        model.fit(X, y)
        path = artifacts_dir / f"model_seed_{seed}.joblib"
        joblib.dump(model, path)
        final_models.append(path.name)

    meta = {
        "method": "lightgbm-seedblend",
        "threshold": best_t,
        "seeds": list(SEEDS),
        "features": FEATURES,
        "model_files": final_models,
    }
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "lightgbm-seedblend",
        "oof_pr_auc": oof_pr_auc,
        "oof_accuracy": best_acc,
        "best_threshold": best_t,
        "majority_baseline_accuracy": majority_baseline,
        "classification_report": report,
        "train_rows": len(y),
        "positive_rate": float(y.mean()),
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"seeds": SEEDS, "params": LGBM_PARAMS})

    plot_pr_curve(y, oof_blend, plots_dir / "oof_pr_curve.png", "OOF seed blend PR")
    plot_threshold_sweep(y, oof_blend, plots_dir / "threshold_sweep.png", best_threshold=best_t, majority_baseline=majority_baseline)
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"t={best_t:.3f}")

    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"Blend OOF PR-AUC: {oof_pr_auc:.4f}, acc @ t={best_t:.3f}: {best_acc:.4f}")
    print(f"OOF positives: {int(oof_pred.sum())}")
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
