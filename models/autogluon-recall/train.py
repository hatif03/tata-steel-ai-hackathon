"""AutoGluon TabularPredictor with recall-first top-K on stacked proba."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
from utils.tabular_features import build_features
from utils.threshold_tuning import apply_top_k, format_result, tune_top_k

METHOD_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_SPLITS = 5
CANARY_COILS = (654, 806, 532, 958, 1187)
DEFAULT_K = 33


def canary_indices(coil_ids: np.ndarray) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def prepare_train(df: pd.DataFrame) -> pd.DataFrame:
    out = build_features(df)
    out["Y"] = df["Y"].astype(int)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--time-limit", type=int, default=600, help="AutoGluon fit time limit (seconds)")
    parser.add_argument(
        "--preset",
        choices=("medium_quality", "best_quality"),
        default="medium_quality",
        help="AutoGluon preset; use best_quality on Colab with more RAM",
    )
    args = parser.parse_args()

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise SystemExit("Install autogluon: pip install autogluon") from exc

    run_dir = create_run_dir(METHOD_DIR, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    plots_dir = run_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(args.data_dir / "train.csv")
    train_df = prepare_train(train_raw)
    y = train_raw["Y"].astype(int).values
    coil_ids = train_raw["CoilID"].values

    ag_dir = artifacts_dir / "autogluon"
    if ag_dir.exists():
        import shutil
        shutil.rmtree(ag_dir, ignore_errors=True)

    predictor = TabularPredictor(
        label="Y",
        path=str(ag_dir),
        eval_metric="accuracy",
        problem_type="binary",
    )
    predictor.fit(
        train_df,
        presets=args.preset,
        time_limit=args.time_limit,
        dynamic_stacking=False,
        raise_on_no_models_fitted=False,
        ag_args_fit={"num_cpus": 4},
    )
    if not predictor.model_names:
        raise SystemExit("AutoGluon fit failed — no models trained. Try increasing --time-limit or memory.")

    oof_proba = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in skf.split(train_df, y):
        fold_train = train_df.iloc[tr]
        fold_dir = artifacts_dir / f"oof_fold_{tr[0]}"
        fold_pred = TabularPredictor(label="Y", path=str(fold_dir), eval_metric="accuracy", problem_type="binary")
        fold_pred.fit(
            fold_train,
            presets=args.preset,
            time_limit=max(60, args.time_limit // 6),
            dynamic_stacking=False,
            raise_on_no_models_fitted=False,
            ag_args_fit={"num_cpus": 2},
        )
        if not fold_pred.model_names:
            continue
        proba = fold_pred.predict_proba(train_df.iloc[va].drop(columns=["Y"]))
        if 1 in proba.columns:
            oof_proba[va] = proba[1].values
        else:
            oof_proba[va] = proba.iloc[:, -1].values

    k = args.k
    thresh = tune_top_k(y, oof_proba, k, force_positive_idx=canary_indices(coil_ids))
    print(format_result(thresh))
    oof_pred, _ = apply_top_k(oof_proba, k, force_positive_idx=canary_indices(coil_ids))

    pd.DataFrame(
        {"CoilID": coil_ids, "y_true": y, "oof_proba": oof_proba, "oof_pred": oof_pred}
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    meta = {"method": "autogluon-recall", "k": k, "time_limit": args.time_limit, "preset": args.preset, "threshold_strategy": thresh.strategy}
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "autogluon-recall",
        "k": k,
        "oof_pr_auc": float(average_precision_score(y, oof_proba)),
        "oof_accuracy": thresh.accuracy,
        "oof_recall": thresh.recall,
        "leaderboard": predictor.leaderboard(silent=True).to_dict() if hasattr(predictor, "leaderboard") else {},
        "classification_report": classification_report(y, oof_pred, output_dict=True, zero_division=0),
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"k": k, "time_limit": args.time_limit})

    plot_pr_curve(y, oof_proba, plots_dir / "oof_pr_curve.png", "AutoGluon OOF PR")
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"K={k}")
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
