"""Cross-model disagreement analysis for Phase 7."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.threshold_tuning import apply_top_k  # noqa: E402

CANARY_COILS = (654, 806, 532, 958, 1187)
METHODS = {
    "sklearn-recall": ("oof_blend", "models/sklearn-recall/outputs/latest"),
    "lightgbm-recall": ("oof_proba", "models/lightgbm-recall/outputs/latest"),
    "gbm-recall": ("oof_blend", "models/gbm-recall/outputs/latest"),
    "recall-blend": ("oof_blend", "models/recall-blend/outputs/latest"),
}
SECONDARY_K = 26


def load_test(method: str, col: str, base: str) -> pd.DataFrame:
    test = pd.read_csv(ROOT / base / "predictions/test_predictions.csv")
    return test[["CoilID", "proba"]].rename(columns={"proba": method})


def positive_set(df: pd.DataFrame, method: str, k: int) -> set[int]:
    canary_idx = [i for i, c in enumerate(df["CoilID"]) if int(c) in CANARY_COILS]
    pred, _ = apply_top_k(df[method].values, k, force_positive_idx=canary_idx)
    coils = df.loc[pred == 1, "CoilID"].astype(int).tolist()
    return set(coils)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-values", nargs="+", type=int, default=[24, 26, 33])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models/phase8-rethreshold/outputs/disagreement_report.json",
    )
    args = parser.parse_args()

    merged = None
    for method, (_, base) in METHODS.items():
        part = load_test(method, "", base)
        merged = part if merged is None else merged.merge(part, on="CoilID")

    report: dict = {"k_analysis": {}, "union_across_methods": {}, "gbm33_vs_union": {}}
    for k in args.k_values:
        sets = {m: positive_set(merged, m, k) for m in METHODS}
        all_pos = set.union(*sets.values())
        overlap_counts = {}
        for m1 in METHODS:
            for m2 in METHODS:
                if m1 >= m2:
                    continue
                overlap_counts[f"{m1}_vs_{m2}"] = len(sets[m1] & sets[m2])

        exclusive = {}
        for m in METHODS:
            others = set.union(*(sets[o] for o in METHODS if o != m))
            exclusive[m] = sorted(sets[m] - others)

        report["k_analysis"][str(k)] = {
            "positives_per_method": {m: len(s) for m, s in sets.items()},
            "pairwise_overlap": overlap_counts,
            "union_unique_coils": len(all_pos),
            "exclusive_to_one_model": exclusive,
        }
        report["union_across_methods"][str(k)] = sorted(all_pos)

    gbm33 = positive_set(merged, "gbm-recall", 33)
    union26 = set()
    for m in ("lightgbm-recall", "sklearn-recall", "recall-blend"):
        union26 |= positive_set(merged, m, SECONDARY_K)
    union26 |= gbm33
    beyond_gbm33 = sorted(union26 - gbm33)
    report["gbm33_vs_union"] = {
        "gbm33_count": len(gbm33),
        "union_k26_plus_gbm33_count": len(union26),
        "beyond_gbm33": beyond_gbm33,
        "gbm_only_at_33": sorted(gbm33 - union26),
        "pairwise_gbm33_vs_lgb26": len(gbm33 & positive_set(merged, "lightgbm-recall", SECONDARY_K)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    for k in args.k_values:
        u = report["union_across_methods"][str(k)]
        print(f"K={k}: union={len(u)} coils")
    gvu = report["gbm33_vs_union"]
    print(f"gbm33={gvu['gbm33_count']} union+gbm33={gvu['union_k26_plus_gbm33_count']} beyond={gvu['beyond_gbm33']}")


if __name__ == "__main__":
    main()
