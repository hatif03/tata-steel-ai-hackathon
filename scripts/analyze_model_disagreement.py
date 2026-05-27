"""Cross-model disagreement analysis (Phase 7 / Phase 9)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.union_augment import CANARY_COILS, DEFAULT_SECONDARY_METHODS  # noqa: E402
from utils.threshold_tuning import apply_top_k  # noqa: E402

ALL_METHODS = {
    "sklearn-recall": "models/sklearn-recall/outputs/latest",
    "lightgbm-recall": "models/lightgbm-recall/outputs/latest",
    "gbm-recall": "models/gbm-recall/outputs/latest",
    "recall-blend": "models/recall-blend/outputs/latest",
    "rf-smote-v2": "models/rf-smote-v2/outputs/latest",
    "mega-recall-blend": "models/mega-recall-blend/outputs/latest",
    "lightgbm-optuna": "models/lightgbm-optuna/outputs/latest",
    "gbm-mega-blend": "models/gbm-mega-blend/outputs/latest",
    "gbm-recall-optuna": "models/gbm-recall-optuna/outputs/latest",
    "catboost-recall": "models/catboost-recall/outputs/latest",
    "meta-recall-stack": "models/meta-recall-stack/outputs/latest",
    "autogluon-recall": "models/autogluon-recall/outputs/latest",
    "gbm-recall-fullmiss": "models/gbm-recall-fullmiss/outputs/latest",
SECONDARY_K = 26


def load_test(method: str, base: str) -> pd.DataFrame | None:
    path = ROOT / base / "predictions/test_predictions.csv"
    if not path.exists():
        return None
    test = pd.read_csv(path)
    return test[["CoilID", "proba"]].rename(columns={"proba": method})


def positive_set(df: pd.DataFrame, method: str, k: int) -> set[int]:
    canary_idx = [i for i, c in enumerate(df["CoilID"]) if int(c) in CANARY_COILS]
    pred, _ = apply_top_k(df[method].values, k, force_positive_idx=canary_idx)
    return set(df.loc[pred == 1, "CoilID"].astype(int))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-values", nargs="+", type=int, default=[24, 26, 33, 38, 44])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(ALL_METHODS.keys()),
        help="Methods to include (skips missing outputs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models/phase8-rethreshold/outputs/disagreement_report.json",
    )
    args = parser.parse_args()

    merged = None
    loaded_methods: list[str] = []
    for method in args.methods:
        base = ALL_METHODS.get(method)
        if base is None:
            continue
        part = load_test(method, base)
        if part is None:
            continue
        merged = part if merged is None else merged.merge(part, on="CoilID")
        loaded_methods.append(method)

    if merged is None:
        raise SystemExit("No model outputs found — train recall models first.")

    report: dict = {
        "loaded_methods": loaded_methods,
        "k_analysis": {},
        "union_across_methods": {},
        "gbm33_vs_union": {},
    }
    for k in args.k_values:
        sets = {m: positive_set(merged, m, k) for m in loaded_methods}
        all_pos = set.union(*sets.values())
        overlap_counts = {}
        for m1 in loaded_methods:
            for m2 in loaded_methods:
                if m1 >= m2:
                    continue
                overlap_counts[f"{m1}_vs_{m2}"] = len(sets[m1] & sets[m2])

        exclusive = {}
        for m in loaded_methods:
            others = set.union(*(sets[o] for o in loaded_methods if o != m))
            exclusive[m] = sorted(sets[m] - others)

        report["k_analysis"][str(k)] = {
            "positives_per_method": {m: len(s) for m, s in sets.items()},
            "pairwise_overlap": overlap_counts,
            "union_unique_coils": len(all_pos),
            "exclusive_to_one_model": exclusive,
        }
        report["union_across_methods"][str(k)] = sorted(all_pos)

    if "gbm-recall" in loaded_methods:
        gbm33 = positive_set(merged, "gbm-recall", 33)
        union26: set[int] = set(gbm33)
        for m in DEFAULT_SECONDARY_METHODS:
            if m in loaded_methods:
                union26 |= positive_set(merged, m, SECONDARY_K)
        beyond_gbm33 = sorted(union26 - gbm33)
        report["gbm33_vs_union"] = {
            "gbm33_count": len(gbm33),
            "union_k26_plus_gbm33_count": len(union26),
            "beyond_gbm33": beyond_gbm33,
            "gbm_only_at_33": sorted(gbm33 - union26),
        }
        if "lightgbm-recall" in loaded_methods:
            report["gbm33_vs_union"]["pairwise_gbm33_vs_lgb26"] = len(
                gbm33 & positive_set(merged, "lightgbm-recall", SECONDARY_K)
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({len(loaded_methods)} methods)")
    for k in args.k_values:
        u = report["union_across_methods"][str(k)]
        print(f"K={k}: union={len(u)} coils")
    if report.get("gbm33_vs_union"):
        gvu = report["gbm33_vs_union"]
        print(f"gbm33={gvu['gbm33_count']} union+gbm33={gvu['union_k26_plus_gbm33_count']} beyond={gvu['beyond_gbm33']}")


if __name__ == "__main__":
    main()
