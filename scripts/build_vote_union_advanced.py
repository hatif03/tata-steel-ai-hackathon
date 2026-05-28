"""Advanced vote-union variants: high-confidence thresholds, weighted vote, gbm33 hybrid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.threshold_tuning import apply_top_k  # noqa: E402
from utils.vote_union import (  # noqa: E402
    DEFAULT_VOTE_METHODS,
    canary_indices,
    compute_vote_counts,
    load_oof_weights,
    merge_method_probas,
    vote_distribution,
    vote_union_predict,
    write_vote_submission,
)

OUTPUT_DIR = ROOT / "models" / "phase8-rethreshold" / "outputs"


def weighted_vote_predict(
    merged,
    methods: list[str],
    k: int,
    canary_idx: list[int],
    weights: dict[str, float],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(merged)
    scores = np.zeros(n, dtype=float)
    votes = np.zeros(n, dtype=int)
    for method in methods:
        pred, _ = apply_top_k(merged[method].values.astype(float), k, force_positive_idx=canary_idx)
        w = weights[method]
        scores += w * pred.astype(float)
        votes += pred
    pred = (scores >= threshold).astype(int)
    for i in canary_idx:
        pred[i] = 1
    return pred, votes


def gbm33_or_vote_predict(
    merged,
    methods: list[str],
    k: int,
    min_votes: int,
    canary_idx: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if "gbm-recall" not in methods:
        raise ValueError("gbm-recall required for gbm33 hybrid")
    gbm_pred, _ = apply_top_k(
        merged["gbm-recall"].values.astype(float), 33, force_positive_idx=canary_idx
    )
    votes = compute_vote_counts(merged, methods, k, canary_idx)
    vote_pred = vote_union_predict(votes, min_votes, canary_idx)
    pred = np.maximum(gbm_pred.astype(int), vote_pred.astype(int))
    for i in canary_idx:
        pred[i] = 1
    return pred, votes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_VOTE_METHODS))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    merged, loaded = merge_method_probas(ROOT, args.methods)
    canary_idx = canary_indices(merged["CoilID"])
    k = args.k
    votes = compute_vote_counts(merged, loaded, k, canary_idx)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for min_v in (10, 12, 14):
        if min_v > len(loaded):
            continue
        name = f"vote-min-k{k}-v{min_v}"
        pred = vote_union_predict(votes, min_v, canary_idx)
        meta = {
            "variant": "high_confidence",
            "k": k,
            "min_votes": min_v,
            "loaded_methods": loaded,
            "test_positives": int(pred.sum()),
            "vote_distribution": vote_distribution(votes),
        }
        write_vote_submission(args.output_dir / name, name, merged["CoilID"], pred, meta)
        manifest.append({"strategy": name, **meta})
        print(f"Wrote {name}: {meta['test_positives']} positives (votes>={min_v})")

    weights = load_oof_weights(ROOT, loaded)
    for frac in (0.15, 0.20, 0.25):
        name = f"vote-weighted-k{k}-t{frac:g}"
        pred, vote_counts = weighted_vote_predict(merged, loaded, k, canary_idx, weights, frac)
        meta = {
            "variant": "weighted_vote",
            "k": k,
            "score_threshold": frac,
            "weights": weights,
            "test_positives": int(pred.sum()),
            "vote_distribution": vote_distribution(vote_counts),
        }
        write_vote_submission(args.output_dir / name, name, merged["CoilID"], pred, meta)
        manifest.append({"strategy": name, **meta})
        print(f"Wrote {name}: {meta['test_positives']} positives (weighted>={frac})")

    for m in (2, 3):
        name = f"vote-hybrid-gbm33-or-m{m}-k{k}"
        pred, vote_counts = gbm33_or_vote_predict(merged, loaded, k, m, canary_idx)
        meta = {
            "variant": "gbm33_or_vote",
            "k": k,
            "min_votes": m,
            "gbm_anchor_k": 33,
            "loaded_methods": loaded,
            "test_positives": int(pred.sum()),
            "vote_distribution": vote_distribution(vote_counts),
        }
        write_vote_submission(args.output_dir / name, name, merged["CoilID"], pred, meta)
        manifest.append({"strategy": name, **meta})
        print(f"Wrote {name}: {meta['test_positives']} positives (gbm33 OR vote>={m})")

    for band_name, lo, hi in (("low", 2, 6), ("high", 7, len(loaded))):
        name = f"vote-band-k{k}-{band_name}"
        pred = np.zeros(len(votes), dtype=int)
        mask = (votes >= lo) & (votes <= hi)
        pred[mask] = 1
        for i in canary_idx:
            pred[i] = 1
        meta = {
            "variant": "vote_band",
            "k": k,
            "vote_band": [lo, hi],
            "loaded_methods": loaded,
            "test_positives": int(pred.sum()),
            "vote_distribution": vote_distribution(votes),
        }
        write_vote_submission(args.output_dir / name, name, merged["CoilID"], pred, meta)
        manifest.append({"strategy": name, **meta})
        print(f"Wrote {name}: {meta['test_positives']} positives (votes in [{lo},{hi}])")

    (args.output_dir / "vote_union_advanced_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
