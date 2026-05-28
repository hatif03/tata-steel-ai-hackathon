"""Vote-union recall: positive if coil in top-K of >= M model vote pool."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_artifacts import (
    copy_to_latest_summary,
    create_run_dir,
    print_saved_artifacts,
    save_metrics,
    save_run_config,
    write_artifacts_manifest,
)
from utils.vote_union import (
    DEFAULT_VOTE_METHODS,
    canary_indices,
    compute_vote_counts,
    merge_method_probas,
    vote_distribution,
    vote_union_predict,
)

METHOD_DIR = Path(__file__).resolve().parent
DEFAULT_K = 40
DEFAULT_MIN_VOTES = 2


def ensure_model_outputs(method: str, data_dir: Path) -> None:
    pred_path = ROOT / f"models/{method}/outputs/latest/predictions/test_predictions.csv"
    if pred_path.is_file():
        return
    train_script = ROOT / f"models/{method}/train.py"
    predict_script = ROOT / f"models/{method}/predict.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"Missing outputs for {method} and no train.py")
    subprocess.run([sys.executable, str(train_script), "--data-dir", str(data_dir)], check=True, cwd=str(ROOT))
    subprocess.run([sys.executable, str(predict_script), "--data-dir", str(data_dir)], check=True, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--min-votes", type=int, default=DEFAULT_MIN_VOTES)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_VOTE_METHODS))
    parser.add_argument("--ensure-pool", action="store_true", help="Train missing vote-pool models")
    args = parser.parse_args()

    if args.ensure_pool:
        for method in args.methods:
            try:
                ensure_model_outputs(method, args.data_dir)
            except FileNotFoundError:
                print(f"Skipping unavailable pool member: {method}")

    run_dir = create_run_dir(METHOD_DIR, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    predictions_dir = run_dir / "predictions"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    merged, loaded = merge_method_probas(ROOT, args.methods)
    canary_idx = canary_indices(merged["CoilID"])
    votes = compute_vote_counts(merged, loaded, args.k, canary_idx)
    pred = vote_union_predict(votes, args.min_votes, canary_idx)

    test_df = pd.read_csv(args.data_dir / "test.csv")
    pd.DataFrame({"CoilID": test_df["CoilID"], "proba": votes.astype(float), "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame({"CoilID": test_df["CoilID"], "Y": pred}).to_csv(
        predictions_dir / "submission.csv", index=False
    )

    meta = {
        "method": "vote-union-recall",
        "k": args.k,
        "min_votes": args.min_votes,
        "vote_methods": loaded,
        "vote_distribution": vote_distribution(votes),
        "test_positives": int(pred.sum()),
        "best_lb_score": 23.01887,
        "best_config": "vote-union-m2-k40",
    }
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "vote-union-recall",
        "k": args.k,
        "min_votes": args.min_votes,
        "n_models": len(loaded),
        "test_positives": int(pred.sum()),
        "vote_distribution": meta["vote_distribution"],
        "best_lb_score": 23.01887,
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, meta)
    copy_to_latest_summary(METHOD_DIR, run_dir)
    print_saved_artifacts(artifacts_dir)
    print(f"Test positives: {int(pred.sum())} (M>={args.min_votes}, K={args.k}, {len(loaded)} models)")


if __name__ == "__main__":
    main()
