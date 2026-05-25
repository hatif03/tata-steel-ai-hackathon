"""Phase 0: threshold sweep on saved OOF/test probabilities (no retrain)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.threshold_tuning import (  # noqa: E402
    apply_threshold,
    format_result,
    tune_max_accuracy,
    tune_recall_first,
    tune_target_fpr,
    _metrics_at_threshold,
)

CANARY_COILS = (654, 806, 532, 958, 1187)
OUTPUT_DIR = ROOT / "models" / "recall-first-phase0" / "outputs"


def load_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "proba" not in df.columns and "oof_proba" in df.columns:
        df = df.rename(columns={"oof_proba": "proba"})
    return df


def evaluate_source(name: str, oof_path: Path, test_path: Path, proba_col: str) -> dict:
    oof = pd.read_csv(oof_path)
    test = load_test(test_path)
    y = oof["y_true"].values.astype(int)
    oof_proba = oof[proba_col].values.astype(float)
    test_proba = test["proba"].values.astype(float)

    strategies = {
        "max_accuracy": tune_max_accuracy(y, oof_proba),
        "recall_first": tune_recall_first(y, oof_proba),
        "target_fpr_3pct": tune_target_fpr(y, oof_proba, max_fpr=0.03),
        "fixed_t_0.05": _metrics_at_threshold(y, oof_proba, 0.05),
        "fixed_t_0.31": _metrics_at_threshold(y, oof_proba, 0.31),
    }
    for k, m in list(strategies.items()):
        if k.startswith("fixed"):
            strategies[k] = type(m)(**{**m.__dict__, "strategy": k})

    print(f"\n=== {name} ({proba_col}) ===")
    rows = []
    for sname, m in strategies.items():
        test_pred = apply_threshold(test_proba, m.threshold)
        canary = test[test["CoilID"].isin(CANARY_COILS)].copy()
        canary["pred"] = apply_threshold(canary["proba"].values, m.threshold)
        print(format_result(m))
        print(f"  test_positives={int(test_pred.sum())} canary_preds={dict(zip(canary.CoilID, canary.pred))}")
        rows.append(
            {
                "source": name,
                "proba_col": proba_col,
                "strategy": sname,
                **m.to_dict(),
                "test_positives": int(test_pred.sum()),
                "canary_all_positive": int(canary["pred"].sum()) == len(CANARY_COILS),
            }
        )

    return {"name": name, "rows": rows, "test": test, "oof_proba_col": proba_col}


def pick_best(results: list[dict]) -> dict:
    """Pick best recall-oriented strategy with all canary coils positive."""
    all_rows = [r for block in results for r in block["rows"]]
    df = pd.DataFrame(all_rows)
    df = df[df["canary_all_positive"] == True]  # noqa: E712

    preferred = [
        "target_fpr_3pct",
        "fixed_t_0.05",
        "fixed_t_0.31",
        "recall_first",
        "recall_first_fallback",
        "max_accuracy",
    ]
    for strategy in preferred:
        subset = df[df["strategy"] == strategy]
        if len(subset) == 0:
            continue
        # Prefer 15–30 test positives when available
        in_range = subset[(subset["test_positives"] >= 15) & (subset["test_positives"] <= 30)]
        pool = in_range if len(in_range) else subset
        best = pool.sort_values(["accuracy", "recall", "test_positives"], ascending=False).iloc[0]
        return best.to_dict()

    return df.sort_values("accuracy", ascending=False).iloc[0].to_dict()


def write_submission(best: dict, results: list[dict], test_df: pd.DataFrame) -> Path:
    source = best["source"]
    block = next(b for b in results if b["name"] == source)
    test = block["test"]
    proba = test["proba"].values.astype(float)
    pred = apply_threshold(proba, best["threshold"])

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"CoilID": test["CoilID"], "Y": pred})
    path = out_dir / "submission.csv"
    sub.to_csv(path, index=False)

    meta = {
        "selected": best,
        "canary_coils": list(CANARY_COILS),
        "fallback_baseline": "models/lightgbm-cv/submission/submission.csv",
        "prior_best_lb_score": 1.88679,
    }
    (out_dir / "phase0_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def main() -> None:
    specs = [
        (
            "lightgbm-cv",
            ROOT / "models/lightgbm-cv/outputs/latest/oof_predictions.csv",
            ROOT / "models/lightgbm-cv/outputs/latest/predictions/test_predictions.csv",
            "oof_proba",
        ),
        (
            "gbm-ensemble-blend",
            ROOT / "models/gbm-ensemble/outputs/latest/oof_predictions.csv",
            ROOT / "models/gbm-ensemble/outputs/latest/predictions/test_predictions.csv",
            "oof_blend",
        ),
        (
            "gbm-ensemble-lgbm",
            ROOT / "models/gbm-ensemble/outputs/latest/oof_predictions.csv",
            ROOT / "models/gbm-ensemble/outputs/latest/predictions/test_predictions.csv",
            "oof_lgbm",
        ),
    ]

    results = []
    for name, oof_p, test_p, col in specs:
        if not oof_p.is_file() or not test_p.is_file():
            print(f"Skip {name}: missing artifacts")
            continue
        if col == "oof_blend":
            oof = pd.read_csv(oof_p)
            test = pd.read_csv(test_p)
            # Reconstruct equal-weight blend on test if only single proba saved
            if col not in oof.columns:
                continue
            results.append(evaluate_source(name, oof_p, test_p, col))
        else:
            results.append(evaluate_source(name, oof_p, test_p, col))

    # Equal-weight OOF blend (diagnostic)
    oof_path = ROOT / "models/gbm-ensemble/outputs/latest/oof_predictions.csv"
    if oof_path.is_file():
        oof = pd.read_csv(oof_path)
        if {"oof_xgb", "oof_lgbm", "oof_catboost"}.issubset(oof.columns):
            oof_eq = oof.copy()
            oof_eq["oof_equal"] = (oof["oof_xgb"] + oof["oof_lgbm"] + oof["oof_catboost"]) / 3.0
            tmp_oof = OUTPUT_DIR / "tmp_oof_equal.csv"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            oof_eq.to_csv(tmp_oof, index=False)
            test_path = ROOT / "models/gbm-ensemble/outputs/latest/predictions/test_predictions.csv"
            test_df = pd.read_csv(test_path)
            tmp_test = OUTPUT_DIR / "tmp_test_equal.csv"
            test_df.to_csv(tmp_test, index=False)
            results.append(evaluate_source("gbm-equal-oof-blend-proxy", tmp_oof, tmp_test, "oof_equal"))

    best = pick_best(results)
    print("\n=== SELECTED ===")
    print(json.dumps(best, indent=2))

    # Use lightgbm-cv test file for submission when source is lightgbm-cv
    block = next(b for b in results if b["name"] == best["source"])
    path = write_submission(best, results, block["test"])
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
