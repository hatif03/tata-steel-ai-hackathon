"""Regenerate vote-union submission from saved meta and vote pool probas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_artifacts import copy_to_latest_summary, resolve_run_dir, save_json
from utils.vote_union import (
    canary_indices,
    compute_vote_counts,
    merge_method_probas,
    vote_union_predict,
)

METHOD_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(METHOD_DIR, args.run_dir)
    meta = joblib.load(run_dir / "artifacts" / "meta.joblib")
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    methods = meta.get("vote_methods", meta.get("methods", []))
    k = int(meta["k"])
    min_votes = int(meta["min_votes"])

    merged, loaded = merge_method_probas(ROOT, methods)
    canary_idx = canary_indices(merged["CoilID"])
    votes = compute_vote_counts(merged, loaded, k, canary_idx)
    pred = vote_union_predict(votes, min_votes, canary_idx)

    test_df = pd.read_csv(args.data_dir / "test.csv")
    pd.DataFrame({"CoilID": test_df["CoilID"], "proba": votes.astype(float), "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame({"CoilID": test_df["CoilID"], "Y": pred}).to_csv(
        predictions_dir / "submission.csv", index=False
    )
    save_json(
        predictions_dir / "predict_meta.json",
        {"test_positives": int(pred.sum()), "k": k, "min_votes": min_votes, "n_models": len(loaded)},
    )
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print(f"Test positives: {int(pred.sum())} / {len(pred)}")


if __name__ == "__main__":
    main()
