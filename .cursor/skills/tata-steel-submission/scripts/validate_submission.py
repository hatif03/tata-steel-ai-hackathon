#!/usr/bin/env python3
"""Validate Tata Steel hackathon submission CSV against test IDs."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TEST = ROOT / "dataset" / "test.csv"


def validate(submission_path: Path, test_path: Path) -> list[str]:
    errors: list[str] = []

    if not submission_path.is_file():
        return [f"Submission file not found: {submission_path}"]

    if not test_path.is_file():
        return [f"Test file not found: {test_path}"]

    sub = pd.read_csv(submission_path)
    test = pd.read_csv(test_path)

    expected_cols = ["CoilID", "Y"]
    if list(sub.columns) != expected_cols:
        errors.append(f"Columns must be exactly {expected_cols}; got {list(sub.columns)}")

    if len(sub) != len(test):
        errors.append(f"Row count must be {len(test)}; got {len(sub)}")

    if sub["CoilID"].duplicated().any():
        dupes = sub.loc[sub["CoilID"].duplicated(), "CoilID"].head(5).tolist()
        errors.append(f"Duplicate CoilID values (first 5): {dupes}")

    test_ids = set(test["CoilID"])
    sub_ids = set(sub["CoilID"])
    missing = test_ids - sub_ids
    extra = sub_ids - test_ids
    if missing:
        errors.append(f"Missing test CoilIDs ({len(missing)}): {sorted(missing)[:5]} ...")
    if extra:
        errors.append(f"Extra CoilIDs not in test ({len(extra)}): {sorted(extra)[:5]} ...")

    if "Y" in sub.columns:
        invalid = sub[~sub["Y"].isin([0, 1])]
        if len(invalid):
            errors.append(f"Y must be 0 or 1; {len(invalid)} invalid rows (e.g. {invalid['Y'].head(3).tolist()})")
        non_int = sub[sub["Y"] != sub["Y"].astype(int)]
        if len(non_int):
            errors.append(f"Y must be integers; {len(non_int)} non-integer values")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Tata Steel submission CSV")
    parser.add_argument("submission", type=Path, help="Path to submission.csv")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST, help="Path to test.csv")
    args = parser.parse_args()

    errors = validate(args.submission, args.test)
    if errors:
        print("VALIDATION FAILED")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    sub = pd.read_csv(args.submission)
    pos = int((sub["Y"] == 1).sum())
    print("VALIDATION OK")
    print(f"  Rows: {len(sub)}")
    print(f"  Positive predictions (Y=1): {pos} ({100 * pos / len(sub):.2f}%)")
    sys.exit(0)


if __name__ == "__main__":
    main()
