"""Probe single engineered ratio features against gbm-recall baseline with canary guard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tabular_features import build_features, feature_names, to_frame

CANARY_GUARD_COILS = (806, 1187)
RANDOM_STATE = 42
N_SPLITS = 5
BLEND_WEIGHTS = {"xgb": 1 / 3, "lgbm": 1 / 3, "catboost": 1 / 3}

PROBE_FEATURES = {
    "ratio_X13_X10": lambda df: df["X13"] / df["X10"].replace(0, np.nan),
    "ratio_X30_X32": lambda df: df["X30"] / df["X32"].replace(0, np.nan),
}

XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
)
LGBM_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    objective="binary", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)
CATBOOST_PARAMS = dict(
    iterations=500, depth=4, learning_rate=0.05, subsample=0.8, random_seed=RANDOM_STATE,
    verbose=0, allow_writing_files=False, auto_class_weights="Balanced",
)


def clone_model(name: str, scale_pos_weight: float):
    if name == "xgb":
        return XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    if name == "lgbm":
        return LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=scale_pos_weight)
    return CatBoostClassifier(**CATBOOST_PARAMS)


def baseline_floor() -> dict[int, float]:
    base = pd.read_csv(ROOT / "models/gbm-recall/outputs/latest/predictions/test_predictions.csv")
    return {int(c): float(base.loc[base["CoilID"] == c, "proba"].iloc[0]) for c in CANARY_GUARD_COILS}


def add_feature(df: pd.DataFrame, name: str) -> pd.DataFrame:
    X = build_features(df)
    X[name] = PROBE_FEATURES[name](df)
    return X


def train_blend_oof(X: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, dict]:
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = {n: np.zeros(len(y)) for n in BLEND_WEIGHTS}
    for tr, va in skf.split(X, y):
        for name in BLEND_WEIGHTS:
            model = clone_model(name, scale_pos_weight)
            model.fit(X.iloc[tr], y[tr])
            oof[name][va] = model.predict_proba(X.iloc[va])[:, 1]
    blend = sum(BLEND_WEIGHTS[n] * oof[n] for n in BLEND_WEIGHTS)
    return blend, oof


def test_blend_proba(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = {n: clone_model(n, scale_pos_weight) for n in BLEND_WEIGHTS}
    for m in models.values():
        m.fit(X_train, y_train)
    return (
        BLEND_WEIGHTS["xgb"] * models["xgb"].predict_proba(X_test)[:, 1]
        + BLEND_WEIGHTS["lgbm"] * models["lgbm"].predict_proba(X_test)[:, 1]
        + BLEND_WEIGHTS["catboost"] * models["catboost"].predict_proba(X_test)[:, 1]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "models/phase8-rethreshold/outputs/feature_probe_report.json")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    y = train["Y"].astype(int).values
    floor = baseline_floor()

    X_base = to_frame(train)
    X_test_base = to_frame(test)
    base_oof, _ = train_blend_oof(X_base, y)
    base_pr = float(average_precision_score(y, base_oof))
    base_test = test_blend_proba(X_base, y, X_test_base)

    results: list[dict] = [
        {
            "feature": "baseline",
            "oof_pr_auc": base_pr,
            "canary_proba": {str(c): float(base_test[np.where(test["CoilID"] == c)[0][0]]) for c in CANARY_GUARD_COILS},
            "passes_canary_guard": True,
        }
    ]

    for fname in PROBE_FEATURES:
        X = add_feature(train, fname)
        X_test = add_feature(test, fname)
        oof, _ = train_blend_oof(X, y)
        pr = float(average_precision_score(y, oof))
        test_proba = test_blend_proba(X, y, X_test)
        canary = {}
        guard_ok = True
        for coil in CANARY_GUARD_COILS:
            idx = np.where(test["CoilID"].values == coil)[0]
            p = float(test_proba[idx[0]])
            canary[str(coil)] = p
            if p < floor[coil]:
                guard_ok = False
        results.append(
            {
                "feature": fname,
                "oof_pr_auc": pr,
                "oof_pr_auc_delta": pr - base_pr,
                "canary_proba": canary,
                "canary_floor": {str(k): v for k, v in floor.items()},
                "passes_canary_guard": guard_ok,
                "recommend": guard_ok and pr >= base_pr,
            }
        )
        status = "KEEP" if guard_ok and pr >= base_pr else "SKIP"
        print(f"{fname}: PR-AUC={pr:.4f} delta={pr - base_pr:+.4f} guard={guard_ok} -> {status}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
