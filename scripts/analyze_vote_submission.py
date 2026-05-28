"""Analyze vote counts for a vote-union submission vs reference methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.vote_union import (  # noqa: E402
    DEFAULT_VOTE_METHODS,
    canary_indices,
    compute_vote_counts,
    merge_method_probas,
)


def load_submission_positives(path: Path) -> set[int]:
    sub = pd.read_csv(path)
    return set(sub.loc[sub["Y"] == 1, "CoilID"].astype(int))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "models/phase8-rethreshold/outputs/vote-union-m2-k40/submission.csv",
    )
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "models/phase8-rethreshold/outputs/union-gbm33-plus-16/submission.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models/phase8-rethreshold/outputs/vote_analysis_m2_k40.json",
    )
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_VOTE_METHODS))
    args = parser.parse_args()

    merged, loaded = merge_method_probas(ROOT, args.methods)
    canary_idx = canary_indices(merged["CoilID"])
    votes = compute_vote_counts(merged, loaded, args.k, canary_idx)

    pos_sub = load_submission_positives(args.submission)
    pos_ref = load_submission_positives(args.reference) if args.reference.is_file() else set()

    rows: list[dict] = []
    for i, coil in enumerate(merged["CoilID"].astype(int)):
        v = int(votes[i])
        in_sub = coil in pos_sub
        in_ref = coil in pos_ref
        if in_sub or v >= 2:
            rows.append({
                "CoilID": coil,
                "votes": v,
                "in_submission": in_sub,
                "in_union_plus16": in_ref,
                "exclusive_vs_plus16": in_sub and not in_ref,
            })

    rows.sort(key=lambda r: (-r["votes"], -int(r["in_submission"])))

    by_vote: dict[str, dict] = {}
    for v in range(0, len(loaded) + 1):
        tier = [r for r in rows if r["votes"] == v]
        if not tier:
            continue
        in_sub = sum(1 for r in tier if r["in_submission"])
        by_vote[str(v)] = {
            "coils": len(tier),
            "in_m2_k40_submission": in_sub,
            "sample_coils": [r["CoilID"] for r in tier[:8]],
        }

    report = {
        "k": args.k,
        "n_models": len(loaded),
        "submission": str(args.submission),
        "submission_positives": len(pos_sub),
        "reference_positives": len(pos_ref),
        "only_in_submission": sorted(pos_sub - pos_ref),
        "only_in_reference": sorted(pos_ref - pos_sub),
        "by_vote_tier": by_vote,
        "top_coils": rows[:40],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Submission positives: {len(pos_sub)} | Reference: {len(pos_ref)}")
    print(f"Exclusive to submission: {len(pos_sub - pos_ref)} coils")


if __name__ == "__main__":
    main()
