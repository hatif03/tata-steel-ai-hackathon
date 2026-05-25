"""Train tuned LightGBM on the winning lightgbm-cv feature set (no ensemble)."""

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

from utils.plotting import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_fold_scores,
    plot_pr_curve,
    plot_threshold_sweep,
)
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
RANDOM_STATE = 42
N_SPLITS = 5
FEATURES = feature_names()

PARAM_GRID = (
    {
        "label": "cv_baseline",
        "n_estimators": 500,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 10,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
    {
        "label": "more_trees",
        "n_estimators": 1000,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 8,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    },
    {
        "label": "leaves31",
        "n_estimators": 700,
        "max_depth": 5,
        "num_leaves": 31,
        "min_child_samples": 6,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 0.2,
    },
    {
        "label": "conservative",
        "n_estimators": 600,
        "max_depth": 3,
        "num_leaves": 7,
        "min_child_samples": 15,
        "learning_rate": 0.05,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.2,
        "reg_lambda": 0.5,
    },
)


def tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float, int]:
    best_t, best_acc, best_pos = 0.5, 0.0, 0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (proba >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        n_pos = int(pred.sum())
        if acc > best_acc or (abs(acc - best_acc) < 1e-9 and n_pos > best_pos):
            best_t, best_acc, best_pos = float(t), float(acc), n_pos
    return best_t, best_acc, best_pos


def lgbm_from_params(scale_pos_weight: float, params: dict) -> LGBMClassifier:
    keys = (
        "n_estimators",
        "max_depth",
        "num_leaves",
        "min_child_samples",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    )
    kw = {k: params[k] for k in keys}
    return LGBMClassifier(
        **kw,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=scale_pos_weight,
    )


def run_cv(X: pd.DataFrame, y: np.ndarray, scale_pos_weight: float, params: dict) -> np.ndarray:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X, y):
        model = lgbm_from_params(scale_pos_weight, params)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof_proba[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    return oof_proba


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tuned LightGBM on base features")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or create_run_dir(METHOD_DIR, args.run_id)
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

    best_params: dict | None = None
    best_oof = np.zeros(len(y))
    best_pr, best_t, best_acc, best_pos = -1.0, 0.5, 0.0, 0

    for candidate in PARAM_GRID:
        oof_proba = run_cv(X, y, scale_pos_weight, candidate)
        oof_pr = float(average_precision_score(y, oof_proba))
        t, acc, n_pos = tune_threshold(y, oof_proba)
        print(
            f"{candidate['label']}: PR-AUC={oof_pr:.4f}, acc={acc:.4f}, "
            f"t={t:.2f}, oof_pos={n_pos}"
        )
        if acc > best_acc or (abs(acc - best_acc) < 1e-9 and oof_pr > best_pr):
            best_params = candidate
            best_oof = oof_proba
            best_pr, best_t, best_acc, best_pos = oof_pr, t, acc, n_pos

    assert best_params is not None
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_pr_aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        fold_pr_aucs.append(float(average_precision_score(y[va_idx], best_oof[va_idx])))

    oof_pred = (best_oof >= best_t).astype(int)
    report = classification_report(y, oof_pred, output_dict=True, zero_division=0)

    pd.DataFrame(
        {"CoilID": coil_ids, "y_true": y, "oof_proba": best_oof, "oof_pred": oof_pred}
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    final_model = lgbm_from_params(scale_pos_weight, best_params)
    final_model.fit(X, y)

    model_txt = artifacts_dir / "lgbm_model.txt"
    joblib.dump(final_model, artifacts_dir / "model.joblib")
    final_model.booster_.save_model(str(model_txt))

    meta = {
        "method": "lightgbm-tuned",
        "threshold": best_t,
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "selected_params": best_params,
    }
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "lightgbm-tuned",
        "oof_pr_auc": float(average_precision_score(y, best_oof)),
        "oof_accuracy": best_acc,
        "best_threshold": best_t,
        "majority_baseline_accuracy": majority_baseline,
        "fold_pr_auc": fold_pr_aucs,
        "classification_report": report,
        "train_rows": len(y),
        "positive_rate": float(y.mean()),
        "selected_param_label": best_params["label"],
        "oof_positives_predicted": best_pos,
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"selected_params": best_params, "data_dir": str(args.data_dir)})

    plot_pr_curve(y, best_oof, plots_dir / "oof_pr_curve.png", "OOF precision-recall")
    plot_threshold_sweep(
        y, best_oof, plots_dir / "threshold_sweep.png",
        best_threshold=best_t, majority_baseline=majority_baseline,
    )
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"t={best_t:.3f}")
    plot_feature_importance(FEATURES, final_model.feature_importances_, plots_dir / "feature_importance.png")
    plot_fold_scores({"PR-AUC": fold_pr_aucs}, plots_dir / "fold_pr_auc.png", "CV fold PR-AUC")

    copy_to_latest_summary(METHOD_DIR, run_dir)

    print(f"Selected: {best_params['label']}")
    print(f"OOF PR-AUC: {metrics['oof_pr_auc']:.4f}, acc: {best_acc:.4f}, t={best_t:.3f}")
    print(f"OOF positives: {best_pos}")
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
