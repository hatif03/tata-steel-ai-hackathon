#!/usr/bin/env python3
"""Profile Tata Steel hackathon dataset. Run from repo root."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "dataset"


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    features = [c for c in train.columns if c.startswith("X")]

    print("=" * 60)
    print("TATA STEEL DATASET PROFILE")
    print("=" * 60)
    print(f"Train: {train.shape}  Test: {test.shape}")
    print(f"Features: {len(features)}  ID overlap: {len(set(train.CoilID) & set(test.CoilID))}")

    vc = train["Y"].value_counts()
    print("\nTarget distribution:")
    print(vc)
    print(f"Positive rate: {vc.get(1, 0) / len(train):.4f}")
    print(f"Majority baseline accuracy: {vc.max() / len(train):.4f}")

    miss_col = train[features].isnull().sum().sort_values(ascending=False)
    miss_row = train[features].isnull().any(axis=1).sum()
    print(f"\nRows with any missing: {miss_row} ({100 * miss_row / len(train):.1f}%)")
    print("Top missing columns:")
    print(miss_col[miss_col > 0].head(10).to_string())

    pos_miss = train.loc[train["Y"] == 1, features].isnull().any(axis=1).mean()
    neg_miss = train.loc[train["Y"] == 0, features].isnull().any(axis=1).mean()
    print(f"\nMissing-row rate  Y=1: {pos_miss:.3f}  Y=0: {neg_miss:.3f}")

    nunique = train[features].nunique(dropna=True)
    low_var = nunique[nunique <= 1]
    if len(low_var):
        print(f"\nConstant features: {list(low_var.index)}")
    else:
        print("\nNo constant features detected.")

    print("\nTest missing rows:", test[features].isnull().any(axis=1).sum())
    print("Done.")


if __name__ == "__main__":
    main()
