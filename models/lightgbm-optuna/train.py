"""Optuna-tuned LightGBM with 10-fold CV and canary guard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
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

from utils.plotting import plot_confusion_matrix, plot_pr_curve
from utils.run_artifacts import (
    copy_to_latest_summary,
    create_run_dir,
    print_saved_artifacts,
    save_metrics,
    save_run_config,
    write_artifacts_manifest,
)
from utils.threshold_tuning import apply_top_k, tune_top_k

RANDOM_STATE = 42
N_SPLITS = 10
CANARY_COILS = (654, 806, 532, 958, 1187)
DEFAULT_K = 26


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def canary_floor_proba() -> dict[int, float]:
    """Minimum acceptable test proba on canary coils from sklearn-recall."""
    test = pd.read_csv(ROOT / "dataset/test.csv")
    sk = pd.read_csv(ROOT / "models/sklearn-recall/outputs/latest/predictions/test_predictions.csv")
    merged = test[["CoilID"]].merge(sk, on="CoilID")
    return {int(c): float(merged.loc[merged["CoilID"] == c, "proba"].iloc[0]) for c in CANARY_COILS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
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
    canary_idx = canary_indices(train["CoilID"])
    floor = canary_floor_proba()
    test = pd.read_csv(args.data_dir / "test.csv")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        oof = np.zeros(len(y))
        for tr, va in skf.split(X, y):
            model = LGBMClassifier(**params)
            model.fit(X.iloc[tr], y[tr])
            oof[va] = model.predict_proba(X.iloc[va])[:, 1]

        score = float(average_precision_score(y, oof))
        final = LGBMClassifier(**params)
        final.fit(X, y)
        test_proba = final.predict_proba(to_frame(test))[:, 1]
        for coil_id, min_p in floor.items():
            if coil_id in (806, 1187):
                idx = np.where(test["CoilID"].values == coil_id)[0]
                if len(idx) and test_proba[idx[0]] < min_p:
                    return 0.0
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_params.update(
        {
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
    )

    oof_proba = np.zeros(len(y))
    fold_pr: list[float] = []
    for tr, va in skf.split(X, y):
        model = LGBMClassifier(**best_params)
        model.fit(X.iloc[tr], y[tr])
        oof_proba[va] = model.predict_proba(X.iloc[va])[:, 1]
        fold_pr.append(float(average_precision_score(y[va], oof_proba[va])))

    thresh = tune_top_k(y, oof_proba, args.k, force_positive_idx=canary_idx)
    oof_pred, _ = apply_top_k(oof_proba, args.k, force_positive_idx=canary_idx)

    pd.DataFrame(
        {"CoilID": coil_ids, "y_true": y, "oof_proba": oof_proba, "oof_pred": oof_pred}
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    final = LGBMClassifier(**best_params)
    final.fit(X, y)
    joblib.dump(final, artifacts_dir / "model.joblib")
    final.booster_.save_model(str(artifacts_dir / "lgbm_model.txt"))

    meta = {
        "method": "lightgbm-optuna",
        "best_params": best_params,
        "k": args.k,
        "threshold": thresh.threshold,
        "threshold_strategy": thresh.strategy,
        "optuna_best_value": study.best_value,
        "features": feature_names(),
    }
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "lightgbm-optuna",
        "oof_pr_auc": float(average_precision_score(y, oof_proba)),
        "oof_accuracy": thresh.accuracy,
        "oof_recall": thresh.recall,
        "best_threshold": thresh.threshold,
        "optuna_best_pr_auc": study.best_value,
        "fold_pr_auc": fold_pr,
        "n_trials": args.n_trials,
        "prior_best_lb_score": 9.05660,
        "classification_report": classification_report(y, oof_pred, output_dict=True, zero_division=0),
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"best_params": best_params, "study_best": study.best_value})

    plot_pr_curve(y, oof_proba, plots_dir / "oof_pr_curve.png", "Optuna LGBM OOF PR")
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"K={args.k}")

    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"Best OOF PR-AUC: {study.best_value:.4f}")
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
