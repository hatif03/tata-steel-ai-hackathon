"""Build vote-union submissions: positive if coil in top-K of >= M models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.vote_union import (  # noqa: E402
    DEFAULT_VOTE_METHODS,
    canary_indices,
    compute_vote_counts,
    merge_method_probas,
    vote_distribution,
    vote_union_predict,
    write_vote_submission,
)

OUTPUT_DIR = ROOT / "models" / "phase8-rethreshold" / "outputs"
DEFAULT_K_VALUES = (38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 55)
DEFAULT_MIN_VOTES = (1, 2, 3, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_VOTE_METHODS))
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--min-votes", nargs="+", type=int, default=list(DEFAULT_MIN_VOTES))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    merged, loaded = merge_method_probas(ROOT, args.methods)
    canary_idx = canary_indices(merged["CoilID"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for k in args.k_values:
        votes = compute_vote_counts(merged, loaded, k, canary_idx)
        for m in args.min_votes:
            if m > len(loaded):
                continue
            name = f"vote-union-m{m}-k{k}"
            pred = vote_union_predict(votes, m, canary_idx)
            meta = {
                "loaded_methods": loaded,
                "methods": loaded,
                "k": k,
                "min_votes": m,
                "test_positives": int(pred.sum()),
                "vote_distribution": vote_distribution(votes),
            }
            write_vote_submission(args.output_dir / name, name, merged["CoilID"], pred, meta)
            manifest.append({"strategy": name, **meta})
            print(f"Wrote {name}: {meta['test_positives']} positives ({len(loaded)} models, M>={m}, K={k})")

    (args.output_dir / "vote_union_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
