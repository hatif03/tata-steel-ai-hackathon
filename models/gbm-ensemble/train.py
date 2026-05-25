"""Train XGBoost + LightGBM + CatBoost ensemble with OOF weight tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, average_precision_score, classification_report
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plotting import (  # noqa: E402
    plot_confusion_matrix,
    plot_fold_scores,
    plot_pr_curve,
    plot_threshold_sweep,
)
from utils.run_artifacts import (  # noqa: E402
    copy_to_latest_summary,
    create_run_dir,
    print_saved_artifacts,
    save_metrics,
    save_run_config,
    write_artifacts_manifest,
)
from utils.tabular_features import feature_names, to_frame  # noqa: E402

METHOD_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_SPLITS = 5
FEATURES = feature_names()
MODEL_NAMES = ("xgb", "lgbm", "catboost")

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

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

CATBOOST_PARAMS = dict(
    iterations=500,
    depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_seed=RANDOM_STATE,
    verbose=0,
    allow_writing_files=False,
    auto_class_weights="Balanced",
)


def make_models(scale_pos_weight: float) -> dict[str, object]:
    return {
        "xgb": XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight),
        "lgbm": LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=scale_pos_weight),
        "catboost": CatBoostClassifier(**CATBOOST_PARAMS),
    }


def clone_model(name: str, scale_pos_weight: float) -> object:
    return make_models(scale_pos_weight)[name]


def tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (proba >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_t, best_acc = float(t), float(acc)
    return best_t, best_acc


def blend_proba(oof: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    out = np.zeros(len(next(iter(oof.values()))))
    for name in MODEL_NAMES:
        out += weights[name] * oof[name]
    return out


def tune_blend_and_threshold(
    y: np.ndarray, oof: dict[str, np.ndarray]
) -> tuple[dict[str, float], float, float]:
    best_weights = {name: 1.0 / len(MODEL_NAMES) for name in MODEL_NAMES}
    best_t, best_acc = tune_threshold(y, blend_proba(oof, best_weights))

    step = 0.05
    grid = np.arange(0.0, 1.0 + step / 2, step)
    for w_xgb in grid:
        for w_lgbm in grid:
            w_cat = 1.0 - w_xgb - w_lgbm
            if w_cat < -1e-9:
                continue
            weights = {"xgb": w_xgb, "lgbm": w_lgbm, "catboost": max(w_cat, 0.0)}
            total = sum(weights.values())
            if total <= 0:
                continue
            weights = {k: v / total for k, v in weights.items()}
            blended = blend_proba(oof, weights)
            t, acc = tune_threshold(y, blended)
            if acc > best_acc:
                best_weights, best_t, best_acc = weights, t, acc

    return best_weights, best_t, best_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GBM ensemble and save run outputs")
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

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = {name: np.zeros(len(y)) for name in MODEL_NAMES}
    fold_pr_aucs: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        for name in MODEL_NAMES:
            model = clone_model(name, scale_pos_weight)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            oof[name][va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]

        equal_w = {n: 1.0 / len(MODEL_NAMES) for n in MODEL_NAMES}
        fold_blend = blend_proba({n: oof[n][va_idx] for n in MODEL_NAMES}, equal_w)
        fold_pr = average_precision_score(y[va_idx], fold_blend)
        fold_pr_aucs.append(float(fold_pr))
        print(f"fold {fold} blend PR-AUC (equal weights): {fold_pr:.4f}")

    blend_weights, best_t, best_acc = tune_blend_and_threshold(y, oof)
    blend_weights = {k: float(v) for k, v in blend_weights.items()}
    oof_blend = blend_proba(oof, blend_weights)
    oof_pr_auc = float(average_precision_score(y, oof_blend))
    oof_pred = (oof_blend >= best_t).astype(int)
    report = classification_report(y, oof_pred, output_dict=True, zero_division=0)

    pd.DataFrame(
        {
            "CoilID": coil_ids,
            "y_true": y,
            "oof_xgb": oof["xgb"],
            "oof_lgbm": oof["lgbm"],
            "oof_catboost": oof["catboost"],
            "oof_blend": oof_blend,
            "oof_pred": oof_pred,
        }
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    final_models: dict[str, object] = {}
    for name in MODEL_NAMES:
        model = clone_model(name, scale_pos_weight)
        model.fit(X, y)
        final_models[name] = model

    final_models["xgb"].save_model(str(artifacts_dir / "xgb_model.json"))
    joblib.dump(final_models["lgbm"], artifacts_dir / "lgbm_model.joblib")
    final_models["lgbm"].booster_.save_model(str(artifacts_dir / "lgbm_model.txt"))
    final_models["catboost"].save_model(str(artifacts_dir / "catboost_model.cbm"))
    joblib.dump(final_models["catboost"], artifacts_dir / "catboost_model.joblib")

    meta = {
        "method": "gbm-ensemble",
        "threshold": best_t,
        "blend_weights": blend_weights,
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "n_splits": N_SPLITS,
        "model_files": {
            "xgb": "xgb_model.json",
            "lgbm": "lgbm_model.joblib",
            "catboost": "catboost_model.cbm",
            "meta": "meta.joblib",
        },
    }
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    base_metrics = {}
    for name in MODEL_NAMES:
        t_solo, acc_solo = tune_threshold(y, oof[name])
        base_metrics[f"oof_{name}_pr_auc"] = float(average_precision_score(y, oof[name]))
        base_metrics[f"oof_{name}_accuracy"] = acc_solo
        base_metrics[f"oof_{name}_threshold"] = t_solo

    metrics = {
        "method": "gbm-ensemble",
        "oof_pr_auc": oof_pr_auc,
        "oof_accuracy": best_acc,
        "best_threshold": best_t,
        "blend_weights": blend_weights,
        "majority_baseline_accuracy": majority_baseline,
        "fold_pr_auc": fold_pr_aucs,
        "classification_report": report,
        "train_rows": len(y),
        "positive_rate": float(y.mean()),
        **base_metrics,
    }
    save_metrics(run_dir, metrics)
    save_run_config(
        run_dir,
        {
            "data_dir": str(args.data_dir.resolve()),
            "run_dir": str(run_dir.resolve()),
            "blend_weights": blend_weights,
            "hyperparameters": {
                "xgb": {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight},
                "lgbm": {**LGBM_PARAMS, "scale_pos_weight": scale_pos_weight},
                "catboost": CATBOOST_PARAMS,
            },
        },
    )

    plot_pr_curve(y, oof_blend, plots_dir / "oof_pr_curve.png", "OOF blend precision-recall")
    plot_threshold_sweep(
        y,
        oof_blend,
        plots_dir / "threshold_sweep.png",
        best_threshold=best_t,
        majority_baseline=majority_baseline,
    )
    plot_confusion_matrix(
        y,
        oof_pred,
        plots_dir / "confusion_matrix_oof.png",
        f"OOF confusion matrix (t={best_t:.3f})",
    )
    plot_fold_scores({"PR-AUC": fold_pr_aucs}, plots_dir / "fold_pr_auc.png", "CV fold blend PR-AUC")

    copy_to_latest_summary(METHOD_DIR, run_dir)

    print(f"Blend weights: {blend_weights}")
    print(f"OOF PR-AUC (blend): {oof_pr_auc:.4f}")
    print(f"OOF accuracy @ t={best_t:.3f}: {best_acc:.4f}")
    print(f"OOF positives predicted: {int(oof_pred.sum())}")
    print(f"Run saved to: {run_dir}")
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
