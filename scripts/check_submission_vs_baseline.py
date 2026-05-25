"""Compare a submission against lightgbm-cv baseline canary coils and positive count."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANARY_COILS = (654, 806, 532, 958, 1187)
BASELINE_TEST_PROBA = ROOT / "models/lightgbm-cv/outputs/latest/predictions/test_predictions.csv"
DEFAULT_MIN_POSITIVES = 10
DEFAULT_MAX_POSITIVES = 35


def load_submission(path: Path) -> pd.DataFrame:
    sub = pd.read_csv(path)
    if list(sub.columns) != ["CoilID", "Y"]:
        raise ValueError(f"Expected columns CoilID,Y; got {list(sub.columns)}")
    return sub


def load_proba(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "proba" not in df.columns:
        return None
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-upload guard vs lightgbm-cv baseline")
    parser.add_argument("submission", type=Path, help="Path to submission.csv")
    parser.add_argument("--proba", type=Path, default=None, help="Optional test_predictions.csv")
    parser.add_argument("--min-positives", type=int, default=DEFAULT_MIN_POSITIVES)
    parser.add_argument("--max-positives", type=int, default=DEFAULT_MAX_POSITIVES)
    args = parser.parse_args()

    sub = load_submission(args.submission)
    proba_path = args.proba
    if proba_path is None:
        sibling = args.submission.parent / "test_predictions.csv"
        if sibling.is_file():
            proba_path = sibling
        else:
            run_pred = args.submission.parent.parent / "predictions" / "test_predictions.csv"
            if run_pred.is_file():
                proba_path = run_pred

    n_pos = int(sub["Y"].sum())
    warnings: list[str] = []
    errors: list[str] = []

    if n_pos < args.min_positives:
        warnings.append(
            f"Only {n_pos} test positives (min recommended {args.min_positives}). "
            "Recall-first strategy typically needs 15–30."
        )
    if n_pos > args.max_positives:
        warnings.append(f"{n_pos} test positives exceeds max recommended {args.max_positives}.")

    canary = sub[sub["CoilID"].isin(CANARY_COILS)].set_index("CoilID")
    missing = [c for c in CANARY_COILS if c not in canary.index]
    if missing:
        errors.append(f"Missing canary CoilIDs: {missing}")

    for coil in CANARY_COILS:
        if coil in canary.index and int(canary.loc[coil, "Y"]) != 1:
            warnings.append(f"Canary coil {coil} predicted Y=0 (baseline lightgbm-cv had Y=1).")

    baseline_proba = load_proba(BASELINE_TEST_PROBA)
    proba_df = load_proba(proba_path) if proba_path else None
    if baseline_proba is not None and proba_df is not None:
        merged = baseline_proba.merge(proba_df, on="CoilID", suffixes=("_base", "_new"))
        for coil in CANARY_COILS:
            row = merged[merged["CoilID"] == coil]
            if len(row) and row.iloc[0]["proba_new"] < row.iloc[0]["proba_base"] * 0.5:
                warnings.append(
                    f"Coil {coil}: proba dropped vs baseline "
                    f"({row.iloc[0]['proba_base']:.3f} -> {row.iloc[0]['proba_new']:.3f})"
                )

    print(f"Submission: {args.submission}")
    print(f"Test positives: {n_pos} / {len(sub)}")
    print(f"Canary predictions: {canary['Y'].to_dict() if len(canary) else {}}")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    if not warnings:
        print("\nOK: no warnings.")
    else:
        print("\nReview warnings before uploading to HackerEarth.")


if __name__ == "__main__":
    main()
