"""Build GBM-anchored union-augmented submissions (Phase 8A / Phase 9A)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.union_augment import (  # noqa: E402
    DEFAULT_SECONDARY_METHODS,
    build_augmented,
    canary_indices,
    merge_test_probas,
)

OUTPUT_DIR = ROOT / "models" / "phase8-rethreshold" / "outputs"
GBM_BASE_K = 33
DEFAULT_TARGET_KS = (35, 36, 37, 38, 39, 40, 41, 42)


def write_submission(name: str, test: pd.DataFrame, pred, meta: dict) -> Path:
    out = OUTPUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "submission.csv"
    pd.DataFrame({"CoilID": test["CoilID"], "Y": pred}).to_csv(path, index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ks", nargs="+", type=int, default=list(DEFAULT_TARGET_KS))
    parser.add_argument("--base-k", type=int, default=GBM_BASE_K)
    parser.add_argument("--secondary-k", type=int, default=26)
    parser.add_argument(
        "--secondary-methods",
        nargs="+",
        default=list(DEFAULT_SECONDARY_METHODS),
    )
    parser.add_argument(
        "--ranking",
        choices=("max", "mean", "weighted"),
        default="max",
        help="How to rank exclusive candidates beyond gbm base set",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    secondary = tuple(args.secondary_methods)
    test, loaded = merge_test_probas(ROOT, secondary_methods=secondary)
    canary_idx = canary_indices(test["CoilID"])
    score_cols = loaded

    manifest: list[dict] = []
    for target_k in args.target_ks:
        if target_k <= args.base_k:
            continue
        pred, coils, added = build_augmented(
            test,
            "gbm-recall",
            secondary,
            score_cols,
            base_k=args.base_k,
            target_k=target_k,
            secondary_k=args.secondary_k,
            canary_idx=canary_idx,
            ranking=args.ranking,
        )
        name = f"union-gbm33-plus-{target_k - args.base_k}"
        meta = {
            "strategy": name,
            "base_k": args.base_k,
            "target_k": target_k,
            "secondary_k": args.secondary_k,
            "secondary_methods_loaded": loaded,
            "ranking": args.ranking,
            "test_positives": int(pred.sum()),
            "positive_coils": coils,
            "added_exclusives": added,
            "canary_all_positive": all(pred[i] == 1 for i in canary_idx),
        }
        path = write_submission(name, test, pred, meta)
        manifest.append({"name": name, "path": str(path), **meta})
        print(f"Wrote {path} ({meta['test_positives']} positives, added={added})")

    (OUTPUT_DIR / "union_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
