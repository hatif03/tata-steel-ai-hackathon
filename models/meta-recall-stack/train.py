"""Meta-stacker on OOF probas from multiple recall base models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, classification_report
from sklearn.model_selection import StratifiedKFold

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
from utils.threshold_tuning import apply_top_k, rank_average_proba, tune_top_k

METHOD_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
CANARY_COILS = (654, 806, 532, 958, 1187)

SOURCE_SPECS: dict[str, tuple[str, str]] = {
    "sklearn-recall": ("oof_blend", "models/sklearn-recall/outputs/latest/oof_predictions.csv"),
    "lightgbm-recall": ("oof_proba", "models/lightgbm-recall/outputs/latest/oof_predictions.csv"),
    "gbm-recall": ("oof_blend", "models/gbm-recall/outputs/latest/oof_predictions.csv"),
    "recall-blend": ("oof_blend", "models/recall-blend/outputs/latest/oof_predictions.csv"),
    "rf-smote-v2": ("oof_proba", "models/rf-smote-v2/outputs/latest/oof_predictions.csv"),
    "mega-recall-blend": ("oof_blend", "models/mega-recall-blend/outputs/latest/oof_predictions.csv"),
    "lightgbm-optuna": ("oof_proba", "models/lightgbm-optuna/outputs/latest/oof_predictions.csv"),
    "gbm-mega-blend": ("oof_blend", "models/gbm-mega-blend/outputs/latest/oof_predictions.csv"),
    "catboost-recall": ("oof_proba", "models/catboost-recall/outputs/latest/oof_predictions.csv"),
    "xgb-recall": ("oof_proba", "models/xgb-recall/outputs/latest/oof_predictions.csv"),
    "lgb-seedblend-recall": ("oof_proba", "models/lgb-seedblend-recall/outputs/latest/oof_predictions.csv"),
    "smote-stack-recall": ("oof_proba", "models/smote-stack-recall/outputs/latest/oof_predictions.csv"),
    "knn-positive-profile": ("oof_proba", "models/knn-positive-profile/outputs/latest/oof_predictions.csv"),
}

K_CANDIDATES = (20, 24, 26, 28, 30, 33, 35, 38, 40, 42, 45)


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def load_oof_matrix(sources: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    frames = []
    loaded: list[str] = []
    for name in sources:
        col, rel = SOURCE_SPECS[name]
        path = ROOT / rel
        if not path.is_file():
            continue
        oof = pd.read_csv(path)
        frames.append(oof[["CoilID", "y_true", col]].rename(columns={col: name}))
        loaded.append(name)
    if not frames:
        raise FileNotFoundError("No base model OOF files found")
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["CoilID", "y_true"])
    y = merged["y_true"].values.astype(int)
    X = merged[loaded].values.astype(float)
    return merged, y, X, loaded


def eval_at_k(name: str, proba: np.ndarray, y: np.ndarray, canary_idx: list[int], k: int) -> dict:
    m = tune_top_k(y, proba, k, force_positive_idx=canary_idx)
    d = m.to_dict()
    d["meta_strategy"] = name
    d["k"] = k
    d["oof_blend"] = proba
    return d


def optimize_weights(X: np.ndarray, y: np.ndarray, canary_idx: list[int], k: int) -> np.ndarray:
    n = X.shape[1]
    x0 = np.ones(n) / n

    def objective(w: np.ndarray) -> float:
        w = np.maximum(w, 0)
        if w.sum() <= 0:
            return 1.0
        w = w / w.sum()
        pred, _ = apply_top_k(X @ w, k, force_positive_idx=canary_idx)
        return -accuracy_score(y, pred)

    res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 800})
    w = np.maximum(res.x, 0)
    return w / w.sum() if w.sum() > 0 else np.ones(n) / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--sources", nargs="+", default=list(SOURCE_SPECS.keys()))
    parser.add_argument("--k", type=int, default=None, help="Fixed K; default picks best from K_CANDIDATES")
    args = parser.parse_args()

    run_dir = create_run_dir(METHOD_DIR, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    plots_dir = run_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    merged, y, X, loaded = load_oof_matrix(args.sources)
    canary_idx = canary_indices(merged["CoilID"])
    k_list = (args.k,) if args.k else K_CANDIDATES

    candidates: list[dict] = []
    stacker = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")
    stacker.fit(X, y)
    stack_proba = stacker.predict_proba(X)[:, 1]

    for k in k_list:
        n = len(loaded)
        equal_w = np.ones(n) / n
        opt_w = optimize_weights(X, y, canary_idx, k)
        rank_proba = rank_average_proba([X[:, i] for i in range(n)])
        for label, proba in [
            ("equal_weight", X @ equal_w),
            ("weight_opt", X @ opt_w),
            ("stacking", stack_proba),
            ("rank_avg", rank_proba),
        ]:
            candidates.append(eval_at_k(label, proba, y, canary_idx, k))

    best = max(candidates, key=lambda c: (c["accuracy"], c["recall"]))
    k = int(best["k"])
    strategy = best["meta_strategy"]
    oof_blend = best["oof_blend"]
    oof_pred, eff_t = apply_top_k(oof_blend, k, force_positive_idx=canary_idx)

    pd.DataFrame(
        {
            "CoilID": merged["CoilID"],
            "y_true": y,
            **{m: X[:, i] for i, m in enumerate(loaded)},
            "oof_blend": oof_blend,
            "oof_pred": oof_pred,
        }
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    meta = {
        "method": "meta-recall-stack",
        "selected_strategy": strategy,
        "k": k,
        "effective_threshold": eff_t,
        "source_models": loaded,
        "k_candidates": list(k_list),
    }
    if strategy == "weight_opt":
        meta["blend_weights"] = dict(zip(loaded, optimize_weights(X, y, canary_idx, k).tolist()))
    if strategy == "stacking":
        joblib.dump(stacker, artifacts_dir / "stacker.joblib")
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "meta-recall-stack",
        "selected_strategy": strategy,
        "k": k,
        "source_models": loaded,
        "oof_pr_auc": float(average_precision_score(y, oof_blend)),
        "oof_accuracy": float(best["accuracy"]),
        "oof_recall": float(best["recall"]),
        "prior_best_lb_score": 14.33964,
        "classification_report": classification_report(y, oof_pred, output_dict=True, zero_division=0),
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"candidates": [{k: v for k, v in c.items() if k != "oof_blend"} for c in candidates]})

    plot_pr_curve(y, oof_blend, plots_dir / "oof_pr_curve.png", "Meta stack OOF PR")
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"K={k}")
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"Selected: {strategy} K={k} acc={best['accuracy']:.4f}")
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
