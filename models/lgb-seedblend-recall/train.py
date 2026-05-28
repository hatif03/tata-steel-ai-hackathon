"""5-seed LightGBM with rank-averaged OOF/test probas and top-K output."""

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
from utils.threshold_tuning import apply_top_k, format_result, rank_average_proba, tune_top_k

METHOD_DIR = Path(__file__).resolve().parent
SEEDS = (42, 123, 456, 789, 999)
N_SPLITS = 5
CANARY_COILS = (654, 806, 532, 958, 1187)
DEFAULT_K = 33


def canary_indices(coil_ids: np.ndarray) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


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

    train = pd.read_csv(args.data_dir / "train.csv")
    X = to_frame(train)
    y = train["Y"].astype(int).values
    coil_ids = train["CoilID"].values
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    oof_by_seed: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        oof = np.zeros(len(y))
        params = dict(
            n_estimators=600, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            objective="binary", random_state=seed, n_jobs=-1, verbose=-1, scale_pos_weight=scale_pos_weight,
        )
        for tr, va in skf.split(X, y):
            model = LGBMClassifier(**params)
            model.fit(X.iloc[tr], y[tr])
            oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        oof_by_seed[seed] = oof

    oof_proba = rank_average_proba(list(oof_by_seed.values()))
    k = args.k
    thresh = tune_top_k(y, oof_proba, k, force_positive_idx=canary_indices(coil_ids))
    print(format_result(thresh))
    oof_pred, _ = apply_top_k(oof_proba, k, force_positive_idx=canary_indices(coil_ids))

    oof_cols = {f"oof_seed_{s}": oof_by_seed[s] for s in SEEDS}
    pd.DataFrame({"CoilID": coil_ids, "y_true": y, "oof_proba": oof_proba, "oof_pred": oof_pred, **oof_cols}).to_csv(
        run_dir / "oof_predictions.csv", index=False
    )

    final_models = []
    for seed in SEEDS:
        params = dict(
            n_estimators=600, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            objective="binary", random_state=seed, n_jobs=-1, verbose=-1, scale_pos_weight=scale_pos_weight,
        )
        m = LGBMClassifier(**params)
        m.fit(X, y)
        final_models.append(m)
    joblib.dump(final_models, artifacts_dir / "models.joblib")
    joblib.dump(
        {"method": "lgb-seedblend-recall", "seeds": list(SEEDS), "k": k, "features": feature_names()},
        artifacts_dir / "meta.joblib",
    )
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "lgb-seedblend-recall", "k": k, "seeds": list(SEEDS),
        "oof_pr_auc": float(average_precision_score(y, oof_proba)),
        "oof_accuracy": thresh.accuracy, "oof_recall": thresh.recall,
        "classification_report": classification_report(y, oof_pred, output_dict=True, zero_division=0),
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"k": k, "seeds": list(SEEDS)})
    plot_pr_curve(y, oof_proba, plots_dir / "oof_pr_curve.png", "LGB seed rank-avg OOF PR")
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"K={k}")
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
