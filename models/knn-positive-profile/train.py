"""kNN positive-profile: max similarity to train Y=1 manifold, top-K output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, pairwise_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.intel_sklearn import sklearn_fit_context
from utils.plotting import plot_confusion_matrix, plot_pr_curve
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


def positive_profile_score(X_pos: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Higher = more similar to positive manifold (1 - min cosine distance)."""
    if len(X_pos) == 0:
        return np.zeros(len(X_query))
    dist = pairwise_distances(X_query, X_pos, metric="cosine")
    min_dist = dist.min(axis=1)
    return 1.0 - min_dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    run_dir = create_run_dir(METHOD_DIR, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    plots_dir = run_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    X_raw = to_frame(train).values
    y = train["Y"].astype(int).values
    coil_ids = train["CoilID"].values

    oof_proba = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    with sklearn_fit_context(use_gpu=True, cpu_only=args.cpu_only):
        for tr, va in skf.split(X_raw, y):
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_raw[tr])
            X_va = imp.transform(X_raw[va])
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)
            pos_idx = np.where(y[tr] == 1)[0]
            oof_proba[va] = positive_profile_score(X_tr_s[pos_idx], X_va_s)

    k = args.k
    thresh = tune_top_k(y, oof_proba, k, force_positive_idx=canary_indices(coil_ids))
    print(format_result(thresh))
    oof_pred, _ = apply_top_k(oof_proba, k, force_positive_idx=canary_indices(coil_ids))

    pd.DataFrame(
        {"CoilID": coil_ids, "y_true": y, "oof_proba": oof_proba, "oof_pred": oof_pred}
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    with sklearn_fit_context(use_gpu=True, cpu_only=args.cpu_only):
        imp = SimpleImputer(strategy="median")
        X_full = imp.fit_transform(X_raw)
        scaler = StandardScaler()
        X_full_s = scaler.fit_transform(X_full)
    joblib.dump(imp, artifacts_dir / "imputer.joblib")
    joblib.dump(scaler, artifacts_dir / "scaler.joblib")
    joblib.dump(
        {"method": "knn-positive-profile", "k": k, "features": feature_names()},
        artifacts_dir / "meta.joblib",
    )
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "knn-positive-profile", "k": k,
        "oof_pr_auc": float(average_precision_score(y, oof_proba)),
        "oof_accuracy": thresh.accuracy, "oof_recall": thresh.recall,
        "classification_report": classification_report(y, oof_pred, output_dict=True, zero_division=0),
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"k": k})
    plot_pr_curve(y, oof_proba, plots_dir / "oof_pr_curve.png", "kNN profile OOF PR")
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"K={k}")
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
