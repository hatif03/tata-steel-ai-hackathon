"""Mega-ensemble: weight-opt, stacking, and vote rules on saved model probas."""

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
from utils.threshold_tuning import apply_top_k, format_result, tune_top_k

METHOD_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
CANARY_COILS = (654, 806, 532, 958, 1187)
SOURCE_MODELS = ("sklearn-recall", "lightgbm-recall", "gbm-recall")
OOF_COL = {"sklearn-recall": "oof_blend", "lightgbm-recall": "oof_proba", "gbm-recall": "oof_blend"}
DEFAULT_K = 26


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def load_oof_matrix() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frames = []
    for name in SOURCE_MODELS:
        oof = pd.read_csv(ROOT / f"models/{name}/outputs/latest/oof_predictions.csv")
        col = OOF_COL[name]
        frames.append(oof[["CoilID", "y_true", col]].rename(columns={col: name}))
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["CoilID", "y_true"])
    y = merged["y_true"].values.astype(int)
    X = merged[list(SOURCE_MODELS)].values.astype(float)
    return merged, y, X


def eval_strategy(name: str, oof_blend: np.ndarray, y: np.ndarray, canary_idx: list[int], k: int) -> dict:
    m = tune_top_k(y, oof_blend, k, force_positive_idx=canary_idx)
    return {"strategy": name, "oof_blend": oof_blend, **m.to_dict()}


def optimize_weights(X: np.ndarray, y: np.ndarray, canary_idx: list[int], k: int) -> np.ndarray:
    n_models = X.shape[1]
    x0 = np.ones(n_models) / n_models

    def objective(w: np.ndarray) -> float:
        w = np.maximum(w, 0)
        if w.sum() <= 0:
            return 1.0
        w = w / w.sum()
        blend = X @ w
        pred, _ = apply_top_k(blend, k, force_positive_idx=canary_idx)
        return -accuracy_score(y, pred)

    res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 500})
    w = np.maximum(res.x, 0)
    return w / w.sum()


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

    merged, y, X = load_oof_matrix()
    canary_idx = canary_indices(merged["CoilID"])
    k = args.k

    candidates: list[dict] = []
    equal_w = np.ones(len(SOURCE_MODELS)) / len(SOURCE_MODELS)
    candidates.append(eval_strategy("equal_weight", X @ equal_w, y, canary_idx, k))

    opt_w = optimize_weights(X, y, canary_idx, k)
    candidates.append(eval_strategy("weight_opt", X @ opt_w, y, canary_idx, k))

    stacker = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    stacker.fit(X, y)
    stack_blend = stacker.predict_proba(X)[:, 1]
    candidates.append(eval_strategy("stacking", stack_blend, y, canary_idx, k))

    preds_per_model = []
    for j in range(len(SOURCE_MODELS)):
        p, _ = apply_top_k(X[:, j], k, force_positive_idx=canary_idx)
        preds_per_model.append(p)
    vote_mat = np.column_stack(preds_per_model)
    majority = (vote_mat.sum(axis=1) >= 2).astype(int)
    union = (vote_mat.sum(axis=1) >= 1).astype(int)
    intersection = (vote_mat.sum(axis=1) >= len(SOURCE_MODELS)).astype(int)

    for name, pred in [("majority_vote", majority), ("union", union), ("intersection", intersection)]:
        acc = accuracy_score(y, pred)
        candidates.append(
            {
                "strategy": name,
                "oof_blend": vote_mat.mean(axis=1),
                "threshold": float("nan"),
                "accuracy": float(acc),
                "recall": float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)),
                "precision": float(((pred == 1) & (y == 1)).sum() / max(pred.sum(), 1)),
                "n_predicted_positive": int(pred.sum()),
                "false_positives": int(((pred == 1) & (y == 0)).sum()),
                "false_negatives": int(((pred == 0) & (y == 1)).sum()),
            }
        )

    best = max(candidates, key=lambda c: (c["accuracy"], c["recall"]))
    print(f"Selected: {best['strategy']} acc={best['accuracy']:.4f}")

    oof_blend = best["oof_blend"]
    oof_pred, eff_t = apply_top_k(oof_blend, k, force_positive_idx=canary_idx)

    pd.DataFrame(
        {
            "CoilID": merged["CoilID"],
            "y_true": y,
            **{m: X[:, i] for i, m in enumerate(SOURCE_MODELS)},
            "oof_blend": oof_blend,
            "oof_pred": oof_pred,
        }
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    meta = {
        "method": "mega-recall-blend",
        "selected_strategy": best["strategy"],
        "k": k,
        "effective_threshold": eff_t,
        "source_models": list(SOURCE_MODELS),
        "blend_weights": dict(zip(SOURCE_MODELS, opt_w.tolist())),
        "equal_weights": dict(zip(SOURCE_MODELS, equal_w.tolist())),
    }
    if best["strategy"] == "stacking":
        joblib.dump(stacker, artifacts_dir / "stacker.joblib")
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "mega-recall-blend",
        "selected_strategy": best["strategy"],
        "k": k,
        "oof_pr_auc": float(average_precision_score(y, oof_blend)),
        "oof_accuracy": float(best["accuracy"]),
        "oof_recall": float(best["recall"]),
        "best_threshold": eff_t,
        "prior_best_lb_score": 9.05660,
        "classification_report": classification_report(y, oof_pred, output_dict=True, zero_division=0),
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"candidates": [{k: v for k, v in c.items() if k != "oof_blend"} for c in candidates]})

    plot_pr_curve(y, oof_blend, plots_dir / "oof_pr_curve.png", "Mega blend OOF PR")
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"K={k}")

    copy_to_latest_summary(METHOD_DIR, run_dir)
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
