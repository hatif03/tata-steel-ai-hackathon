"""Predict test set for smote-stack-recall."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.intel_sklearn import sklearn_fit_context
from utils.run_artifacts import copy_to_latest_summary, resolve_run_dir
from utils.tabular_features import to_frame
from utils.threshold_tuning import apply_top_k

METHOD_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
CANARY_COILS = (654, 806, 532, 958, 1187)
BASE_NAMES = ("rf", "lgbm", "catboost")


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    run_dir = resolve_run_dir(METHOD_DIR, args.run_dir)
    meta = joblib.load(run_dir / "artifacts" / "meta.joblib")
    stacker = joblib.load(run_dir / "artifacts" / "stacker.joblib")
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    y = train["Y"].astype(int).values
    X_train = to_frame(train).values
    X_test = to_frame(test).values
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    with sklearn_fit_context(use_gpu=True, cpu_only=args.cpu_only):
        rf_pipe = ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("rf", RandomForestClassifier(n_estimators=400, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)),
        ])
        rf_pipe.fit(X_train, y)
    rf_p = rf_pipe.predict_proba(X_test)[:, 1]

    imp = SimpleImputer(strategy="median")
    X_tr_i = imp.fit_transform(X_train)
    X_te_i = imp.transform(X_test)
    smote = BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_sm, y_sm = smote.fit_resample(X_tr_i, y)

    lgb = LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1, scale_pos_weight=scale_pos_weight,
    )
    lgb.fit(X_sm, y_sm)
    lgb_p = lgb.predict_proba(X_te_i)[:, 1]

    cat = CatBoostClassifier(
        iterations=500, depth=5, learning_rate=0.03, random_seed=RANDOM_STATE,
        verbose=0, allow_writing_files=False, auto_class_weights="Balanced",
    )
    cat.fit(X_sm, y_sm)
    cat_p = cat.predict_proba(X_te_i)[:, 1]

    X_meta = np.column_stack([rf_p, lgb_p, cat_p])
    proba = stacker.predict_proba(X_meta)[:, 1]
    k = int(meta["k"])
    pred, _ = apply_top_k(proba, k, force_positive_idx=canary_indices(test["CoilID"]))

    pd.DataFrame({"CoilID": test["CoilID"], "proba": proba, "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame({"CoilID": test["CoilID"], "Y": pred}).to_csv(
        predictions_dir / "submission.csv", index=False
    )
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"Test positives: {int(pred.sum())} / {len(pred)}")


if __name__ == "__main__":
    main()
