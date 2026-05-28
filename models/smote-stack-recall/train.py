"""SMOTE stack: BorderlineSMOTE + CatBoost/LGB/RF bases, LogisticRegression meta, top-K."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import StratifiedKFold

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
BASE_NAMES = ("rf", "lgbm", "catboost")


def canary_indices(coil_ids: np.ndarray) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def make_rf_pipeline(cpu_only: bool) -> ImbPipeline:
    rf = RandomForestClassifier(n_estimators=400, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    return ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("rf", rf),
    ])


def fit_base_oof(X_raw: np.ndarray, y: np.ndarray, cpu_only: bool) -> dict[str, np.ndarray]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = {n: np.zeros(len(y)) for n in BASE_NAMES}
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    for tr, va in skf.split(X_raw, y):
        X_tr, X_va = X_raw[tr], X_raw[va]
        with sklearn_fit_context(use_gpu=True, cpu_only=cpu_only):
            rf_pipe = make_rf_pipeline(cpu_only)
            rf_pipe.fit(X_tr, y[tr])
        oof["rf"][va] = rf_pipe.predict_proba(X_va)[:, 1]

        imp = SimpleImputer(strategy="median")
        X_tr_i = imp.fit_transform(X_tr)
        X_va_i = imp.transform(X_va)
        smote = BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_sm, y_sm = smote.fit_resample(X_tr_i, y[tr])

        lgb = LGBMClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, scale_pos_weight=scale_pos_weight,
        )
        lgb.fit(X_sm, y_sm)
        oof["lgbm"][va] = lgb.predict_proba(X_va_i)[:, 1]

        cat = CatBoostClassifier(
            iterations=500, depth=5, learning_rate=0.03, random_seed=RANDOM_STATE,
            verbose=0, allow_writing_files=False, auto_class_weights="Balanced",
        )
        cat.fit(X_sm, y_sm)
        oof["catboost"][va] = cat.predict_proba(X_va_i)[:, 1]

    return oof


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

    oof_bases = fit_base_oof(X_raw, y, args.cpu_only)
    X_meta = np.column_stack([oof_bases[n] for n in BASE_NAMES])
    stacker = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")
    stacker.fit(X_meta, y)
    oof_proba = stacker.predict_proba(X_meta)[:, 1]

    k = args.k
    thresh = tune_top_k(y, oof_proba, k, force_positive_idx=canary_indices(coil_ids))
    print(format_result(thresh))
    oof_pred, _ = apply_top_k(oof_proba, k, force_positive_idx=canary_indices(coil_ids))

    pd.DataFrame({
        "CoilID": coil_ids, "y_true": y,
        **{f"oof_{n}": oof_bases[n] for n in BASE_NAMES},
        "oof_proba": oof_proba, "oof_pred": oof_pred,
    }).to_csv(run_dir / "oof_predictions.csv", index=False)

    joblib.dump(stacker, artifacts_dir / "stacker.joblib")
    joblib.dump(
        {"method": "smote-stack-recall", "k": k, "base_models": list(BASE_NAMES), "features": feature_names()},
        artifacts_dir / "meta.joblib",
    )
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "smote-stack-recall", "k": k,
        "oof_pr_auc": float(average_precision_score(y, oof_proba)),
        "oof_accuracy": thresh.accuracy, "oof_recall": thresh.recall,
        "classification_report": classification_report(y, oof_pred, output_dict=True, zero_division=0),
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, {"k": k})
    plot_pr_curve(y, oof_proba, plots_dir / "oof_pr_curve.png", "SMOTE stack OOF PR")
    plot_confusion_matrix(y, oof_pred, plots_dir / "confusion_matrix_oof.png", f"K={k}")
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print_saved_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
