"""Union GBM33 augment: gbm-recall top-33 anchor + secondary-model exclusives."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
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
from utils.union_augment import (
    DEFAULT_SECONDARY_METHODS,
    build_augmented,
    canary_indices,
    merge_test_probas,
)

METHOD_DIR = Path(__file__).resolve().parent
CANARY_COILS = (654, 806, 532, 958, 1187)


def ensure_model_outputs(method: str, data_dir: Path) -> None:
    pred_path = ROOT / f"models/{method}/outputs/latest/predictions/test_predictions.csv"
    if pred_path.is_file():
        return
    train_script = ROOT / f"models/{method}/train.py"
    predict_script = ROOT / f"models/{method}/predict.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"Missing outputs for {method} and no train.py")
    subprocess.run(
        [sys.executable, str(train_script), "--data-dir", str(data_dir)],
        check=True,
        cwd=str(ROOT),
    )
    subprocess.run(
        [sys.executable, str(predict_script), "--data-dir", str(data_dir)],
        check=True,
        cwd=str(ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--base-k", type=int, default=33)
    parser.add_argument("--target-k", type=int, default=38)
    parser.add_argument("--secondary-k", type=int, default=26)
    parser.add_argument(
        "--secondary-methods",
        nargs="+",
        default=list(DEFAULT_SECONDARY_METHODS),
    )
    parser.add_argument("--ranking", choices=("max", "mean", "weighted"), default="max")
    parser.add_argument("--train-gbm", action="store_true", help="Retrain gbm-recall if missing")
    parser.add_argument("--ensure-secondaries", action="store_true", help="Train missing secondary models")
    args = parser.parse_args()

    if args.train_gbm:
        ensure_model_outputs("gbm-recall", args.data_dir)

    secondary = tuple(args.secondary_methods)
    if args.ensure_secondaries:
        for method in secondary:
            try:
                ensure_model_outputs(method, args.data_dir)
            except FileNotFoundError:
                print(f"Skipping unavailable secondary: {method}")

    run_dir = create_run_dir(METHOD_DIR, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    predictions_dir = run_dir / "predictions"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    test, loaded = merge_test_probas(ROOT, secondary_methods=secondary)
    canary_idx = canary_indices(test["CoilID"])
    pred, coils, added = build_augmented(
        test,
        "gbm-recall",
        secondary,
        loaded,
        base_k=args.base_k,
        target_k=args.target_k,
        secondary_k=args.secondary_k,
        canary_idx=canary_idx,
        ranking=args.ranking,
    )

    test_df = pd.read_csv(args.data_dir / "test.csv")
    gbm_proba = test["gbm-recall"].values.astype(float)
    pd.DataFrame({"CoilID": test_df["CoilID"], "proba": gbm_proba, "Y": pred}).to_csv(
        predictions_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame({"CoilID": test_df["CoilID"], "Y": pred}).to_csv(
        predictions_dir / "submission.csv", index=False
    )

    canary_mask = test_df["CoilID"].isin(CANARY_COILS)
    meta = {
        "method": "union-gbm33-augment",
        "base_k": args.base_k,
        "target_k": args.target_k,
        "secondary_k": args.secondary_k,
        "secondary_methods": loaded,
        "ranking": args.ranking,
        "added_exclusives": added,
        "positive_coils": coils,
        "canary_predictions": {
            int(k): int(v)
            for k, v in zip(
                test_df.loc[canary_mask, "CoilID"].astype(int),
                pred[canary_mask.to_numpy()],
            )
        },
    }
    joblib.dump(meta, artifacts_dir / "meta.joblib")
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "union-gbm33-augment",
        "test_positives": int(pred.sum()),
        "target_k": args.target_k,
        "added_exclusives": added,
        "prior_best_lb_score": 14.33964,
    }
    save_metrics(run_dir, metrics)
    save_run_config(run_dir, meta)

    copy_to_latest_summary(METHOD_DIR, run_dir)
    print_saved_artifacts(artifacts_dir)
    print(f"Test positives: {int(pred.sum())} (target_k={args.target_k}, added={added})")


if __name__ == "__main__":
    main()
