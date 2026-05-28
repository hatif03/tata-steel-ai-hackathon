"""Batch SHAP-inspired ratio feature probe with canary guard."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_single_feature import (  # noqa: E402
    CANARY_GUARD_COILS,
    baseline_floor,
    test_blend_proba,
    train_blend_oof,
)
from utils.tabular_features import build_features, to_frame

TOP_FEATURES = ("X13", "X10", "X30", "X32", "X15", "X1", "X7", "X22", "X33", "X28")


def add_ratio(df: pd.DataFrame, a: str, b: str, name: str) -> pd.DataFrame:
    X = build_features(df) if "X1" not in df.columns else to_frame(df)
    X[name] = df[a] / df[b].replace(0, np.nan)
    return X


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models/phase8-rethreshold/outputs/shap_ratio_probe_report.json",
    )
    parser.add_argument("--max-pairs", type=int, default=12, help="Limit ratio pairs probed")
    args = parser.parse_args()

    try:
        floor = baseline_floor()
    except FileNotFoundError:
        raise SystemExit("Train gbm-recall first: python models/gbm-recall/train.py && predict.py")

    train = pd.read_csv(ROOT / "dataset/train.csv")
    test = pd.read_csv(ROOT / "dataset/test.csv")
    y = train["Y"].astype(int).values

    pairs = list(itertools.combinations(TOP_FEATURES, 2))[: args.max_pairs]
    results: list[dict] = []

    for a, b in pairs:
        name = f"ratio_{a}_{b}"
        try:
            X_train = add_ratio(train, a, b, name)
            X_test = add_ratio(test, a, b, name)
            oof_blend, _ = train_blend_oof(X_train, y)
            test_proba_arr = test_blend_proba(X_train, y, X_test)
            canary = {}
            guard_ok = True
            for coil in CANARY_GUARD_COILS:
                idx = np.where(test["CoilID"].values == coil)[0]
                p = float(test_proba_arr[idx[0]])
                canary[str(coil)] = p
                if p < floor[coil]:
                    guard_ok = False
            oof_pr = float(average_precision_score(y, oof_blend))
            results.append({
                "feature": name,
                "canary_guard": "PASS" if guard_ok else "SKIP",
                "oof_pr_auc": oof_pr,
                "canary_proba": canary,
            })
            print(f"{name}: {'PASS' if guard_ok else 'SKIP'} OOF PR-AUC={oof_pr:.4f}")
        except Exception as exc:
            results.append({"feature": name, "canary_guard": "ERROR", "error": str(exc)})

    passing = [r for r in results if r.get("canary_guard") == "PASS"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"probed": len(results), "passing": passing, "all": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nPassing features: {len(passing)} / {len(results)}")


if __name__ == "__main__":
    main()
