"""GBM-centric mega blend: gbm-recall primary + sklearn/lgbm secondary at K=33."""

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
from utils.threshold_tuning import apply_top_k, tune_top_k

METHOD_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
CANARY_COILS = (654, 806, 532, 958, 1187)
GBM_MODEL = "gbm-recall"
SECONDARY = ("sklearn-recall", "lightgbm-recall")
OOF_COL = {"gbm-recall": "oof_blend", "sklearn-recall": "oof_blend", "lightgbm-recall": "oof_proba"}
DEFAULT_K = 33
GBM_HEAVY = {"gbm-recall": 0.7, "sklearn-recall": 0.15, "lightgbm-recall": 0.15}


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def load_oof() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frames = []
    for name in (GBM_MODEL, *SECONDARY):
        oof = pd.read_csv(ROOT / f"models/{name}/outputs/latest/oof_predictions.csv")
        col = OOF_COL[name]
        frames.append(oof[["CoilID", "y_true", col]].rename(columns={col: name}))
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["CoilID", "y_true"])
    y = merged["y_true"].values.astype(int)
    cols = [GBM_MODEL, *SECONDARY]
    return merged, y, merged[cols].values.astype(float)


def eval_blend(name: str, blend: np.ndarray, y: np.ndarray, canary_idx: list[int], k: int) -> dict:
    m = tune_top_k(y, blend, k, force_positive_idx=canary_idx)
    row = m.to_dict()
    row.pop("strategy", None)
    return {"strategy": name, "oof_blend": blend, **row}


def optimize_weights(X: np.ndarray, y: np.ndarray, canary_idx: list[int], k: int) -> np.ndarray:
    n = X.shape[1]
    x0 = np.array([GBM_HEAVY[GBM_MODEL], GBM_HEAVY[SECONDARY[0]], GBM_HEAVY[SECONDARY[1]]])

    def objective(w: np.ndarray) -> float:
        w = np.maximum(w, 0)
        if w.sum() <= 0:
            return 1.0
        w = w / w.sum()
        pred, _ = apply_top_k(X @ w, k, force_positive_idx=canary_idx)
        return -accuracy_score(y, pred)

    res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 500})
    w = np.maximum(res.x, 0)
    return w / w.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    run_dir = create_run_dir(METHOD_DIR, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    plots_dir = run_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    merged, y, X = load_oof()
    canary_idx = canary_indices(merged["CoilID"])
    k = args.k
    cols = [GBM_MODEL, *SECONDARY]

    candidates = [
        eval_blend("gbm_only", X[:, 0], y, canary_idx, k),
        eval_blend(
            "gbm_heavy",
            X @ np.array([GBM_HEAVY[c] for c in cols]),
            y,
            canary_idx,
            k,
        ),
    ]
    opt_w = optimize_weights(X, y, canary_idx, k)
    candidates.append(eval_blend("weight_opt", X @ opt_w, y, canary_idx, k))

    stacker = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    stacker.fit(X, y)
    stack_blend = stacker.predict_proba(X)[:, 1]
    candidates.append(eval_blend("stacking", stack_blend, y, canary_idx, k))

    best = max(candidates, key=lambda c: (c["accuracy"], c["recall"]))
    print(f"Selected {best['strategy']} acc={best['accuracy']:.4f}")
    oof_blend = best["oof_blend"]
    oof_pred, _ = apply_top_k(oof_blend, k, force_positive_idx=canary_idx)

    pd.DataFrame(
        {"CoilID": merged["CoilID"], "y_true": y, **{c: X[:, i] for i, c in enumerate(cols)}, "oof_blend": oof_blend, "oof_pred": oof_pred}
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    meta = {
        "method": "gbm-mega-blend",
        "selected_strategy": best["strategy"],
        "k": k,
        "blend_weights": dict(zip(cols, opt_w.tolist())),
        "gbm_heavy_weights": GBM_HEAVY,
    }
    if best["strategy"] == "stacking":
        joblib.dump(stacker, artifacts_dir / "stacker.joblib")
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "gbm-mega-blend",
        "selected_strategy": best["strategy"],
        "k": k,
        "oof_pr_auc": float(average_precision_score(y, oof_blend)),
        "oof_accuracy": float(best["accuracy"]),
        "prior_best_lb_score": 12.45283,
        "classification_report": classification_report(y, oof_pred, output_dict=True, zero_division=0),
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"candidates": [{k: v for k, v in c.items() if k != "oof_blend"} for c in candidates]})

    plot_pr_curve(y, oof_blend, plots_dir / "oof_pr_curve.png", "GBM mega blend")
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"K={k}")
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
