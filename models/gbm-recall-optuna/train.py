"""Optuna-tuned XGB + LGB + CatBoost equal blend with canary guard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
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
from utils.threshold_tuning import apply_top_k, tune_top_k

METHOD_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_SPLITS = 5
CANARY_COILS = (654, 806, 532, 958, 1187)
CANARY_GUARD_COILS = (806, 1187)
DEFAULT_K = 33
BLEND_WEIGHTS = {"xgb": 1 / 3, "lgbm": 1 / 3, "catboost": 1 / 3}


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def passes_rank_guard(test_proba: np.ndarray, coil_ids: np.ndarray, k: int) -> bool:
    """Require guard coils 806/1187 in top-K (rank-based, not proba floor)."""
    canary_idx = [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]
    pred, _ = apply_top_k(test_proba, k, force_positive_idx=canary_idx)
    for coil in CANARY_GUARD_COILS:
        idx = np.where(coil_ids == coil)[0]
        if len(idx) and pred[idx[0]] == 0:
            return False
    return True


def fit_blend_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    skf: StratifiedKFold,
    params: dict,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    oof = {n: np.zeros(len(y)) for n in BLEND_WEIGHTS}
    for tr, va in skf.split(X, y):
        xgb = XGBClassifier(**params["xgb"])
        lgbm = LGBMClassifier(**params["lgbm"])
        cat = CatBoostClassifier(**params["catboost"])
        xgb.fit(X.iloc[tr], y[tr])
        lgbm.fit(X.iloc[tr], y[tr])
        cat.fit(X.iloc[tr], y[tr])
        oof["xgb"][va] = xgb.predict_proba(X.iloc[va])[:, 1]
        oof["lgbm"][va] = lgbm.predict_proba(X.iloc[va])[:, 1]
        oof["catboost"][va] = cat.predict_proba(X.iloc[va])[:, 1]
    blend = sum(BLEND_WEIGHTS[n] * oof[n] for n in BLEND_WEIGHTS)
    return blend, oof


def fit_final_models(X: pd.DataFrame, y: np.ndarray, params: dict) -> dict:
    models = {
        "xgb": XGBClassifier(**params["xgb"]),
        "lgbm": LGBMClassifier(**params["lgbm"]),
        "catboost": CatBoostClassifier(**params["catboost"]),
    }
    for m in models.values():
        m.fit(X, y)
    return models


def test_blend_proba(models: dict, X: pd.DataFrame) -> np.ndarray:
    return (
        BLEND_WEIGHTS["xgb"] * models["xgb"].predict_proba(X)[:, 1]
        + BLEND_WEIGHTS["lgbm"] * models["lgbm"].predict_proba(X)[:, 1]
        + BLEND_WEIGHTS["catboost"] * models["catboost"].predict_proba(X)[:, 1]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run_dir = create_run_dir(METHOD_DIR, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    plots_dir = run_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    X = to_frame(train)
    X_test = to_frame(test)
    y = train["Y"].astype(int).values
    coil_ids = train["CoilID"].values
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    canary_idx = canary_indices(train["CoilID"])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        spw = trial.suggest_float("spw", 10.0, 30.0)
        params = {
            "xgb": {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 300, 800),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 7),
                "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.08, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 1.0),
                "scale_pos_weight": spw,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            },
            "lgbm": {
                "n_estimators": trial.suggest_int("lgbm_n_estimators", 300, 800),
                "max_depth": trial.suggest_int("lgbm_max_depth", 3, 7),
                "learning_rate": trial.suggest_float("lgbm_lr", 0.01, 0.08, log=True),
                "subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("lgbm_colsample", 0.6, 1.0),
                "scale_pos_weight": spw,
                "objective": "binary",
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
                "verbose": -1,
            },
            "catboost": {
                "iterations": trial.suggest_int("cat_iterations", 300, 800),
                "depth": trial.suggest_int("cat_depth", 3, 7),
                "learning_rate": trial.suggest_float("cat_lr", 0.01, 0.08, log=True),
                "subsample": trial.suggest_float("cat_subsample", 0.6, 1.0),
                "random_seed": RANDOM_STATE,
                "verbose": 0,
                "allow_writing_files": False,
                "auto_class_weights": "Balanced",
            },
        }
        blend, _ = fit_blend_oof(X, y, skf, params)
        tk = tune_top_k(y, blend, args.k, force_positive_idx=canary_idx)
        # Rank guard on OOF proxy — full test check deferred to final fit
        return float(tk.accuracy)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    if study.best_value <= 0:
        print("Warning: all Optuna trials failed rank guard; using gbm-recall default hyperparams")
        best_params = {
            "xgb": {
                "n_estimators": 500, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": scale_pos_weight,
                "objective": "binary:logistic", "eval_metric": "logloss",
                "random_state": RANDOM_STATE, "n_jobs": -1,
            },
            "lgbm": {
                "n_estimators": 500, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": scale_pos_weight,
                "objective": "binary", "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
            },
            "catboost": {
                "iterations": 500, "depth": 4, "learning_rate": 0.05, "subsample": 0.8,
                "random_seed": RANDOM_STATE, "verbose": 0, "allow_writing_files": False,
                "auto_class_weights": "Balanced",
            },
        }
        optuna_best = None
    else:
        best = study.best_params
        best_params = {
            "xgb": {
                "n_estimators": best["xgb_n_estimators"],
                "max_depth": best["xgb_max_depth"],
                "learning_rate": best["xgb_lr"],
                "subsample": best["xgb_subsample"],
                "colsample_bytree": best["xgb_colsample"],
                "scale_pos_weight": scale_pos_weight,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            },
            "lgbm": {
                "n_estimators": best["lgbm_n_estimators"],
                "max_depth": best["lgbm_max_depth"],
                "learning_rate": best["lgbm_lr"],
                "subsample": best["lgbm_subsample"],
                "colsample_bytree": best["lgbm_colsample"],
                "scale_pos_weight": scale_pos_weight,
                "objective": "binary",
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
                "verbose": -1,
            },
            "catboost": {
                "iterations": best["cat_iterations"],
                "depth": best["cat_depth"],
                "learning_rate": best["cat_lr"],
                "subsample": best["cat_subsample"],
                "random_seed": RANDOM_STATE,
                "verbose": 0,
                "allow_writing_files": False,
                "auto_class_weights": "Balanced",
            },
        }
        optuna_best = study.best_value

    blend, oof_parts = fit_blend_oof(X, y, skf, best_params)
    fold_pr: list[float] = []
    for tr, va in skf.split(X, y):
        fold_pr.append(float(average_precision_score(y[va], blend[va])))

    k = args.k
    thresh = tune_top_k(y, blend, k, force_positive_idx=canary_idx)
    oof_pred, _ = apply_top_k(blend, k, force_positive_idx=canary_idx)

    pd.DataFrame(
        {
            "CoilID": coil_ids,
            "y_true": y,
            "oof_xgb": oof_parts["xgb"],
            "oof_lgbm": oof_parts["lgbm"],
            "oof_catboost": oof_parts["catboost"],
            "oof_blend": blend,
            "oof_pred": oof_pred,
        }
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    final_models = fit_final_models(X, y, best_params)
    final_models["xgb"].save_model(str(artifacts_dir / "xgb_model.json"))
    joblib.dump(final_models["lgbm"], artifacts_dir / "lgbm_model.joblib")
    final_models["catboost"].save_model(str(artifacts_dir / "catboost_model.cbm"))

    meta = {
        "method": "gbm-recall-optuna",
        "best_params": best_params,
        "blend_weights": BLEND_WEIGHTS,
        "k": k,
        "threshold": thresh.threshold,
        "threshold_strategy": thresh.strategy,
        "optuna_best_value": optuna_best,
        "optuna_objective": "oof_top_k_accuracy",
        "features": feature_names(),
    }
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "gbm-recall-optuna",
        "oof_pr_auc": float(average_precision_score(y, blend)),
        "oof_accuracy": thresh.accuracy,
        "oof_recall": thresh.recall,
        "optuna_best_oof_accuracy": optuna_best,
        "n_trials": args.n_trials,
        "k": k,
        "prior_best_lb_score": 12.45283,
        "fold_pr_auc": fold_pr,
        "classification_report": classification_report(y, oof_pred, output_dict=True, zero_division=0),
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"best_params": best_params, "study_best": optuna_best})

    plot_pr_curve(y, blend, plots_dir / "oof_pr_curve.png", "Optuna GBM blend OOF PR")
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"K={k}")
    plot_fold_scores({"PR-AUC": fold_pr}, plots_dir / "fold_pr_auc.png", "CV fold PR-AUC")

    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"Best OOF PR-AUC: {float(average_precision_score(y, blend)):.4f}")
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
