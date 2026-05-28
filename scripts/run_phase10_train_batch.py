"""Batch train Phase 10 core and secondary recall models."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

CORE_TRAIN_PREDICT = [
    ("gbm-recall", ["train.py", "--threshold-strategy", "top_k_33"], "predict.py"),
    ("lightgbm-recall", ["train.py"], "predict.py"),
    ("sklearn-recall", ["train.py", "--cpu-only"], "predict.py"),
    ("recall-blend", ["train.py"], "predict.py"),
    ("catboost-recall", ["train.py"], "predict.py"),
    ("rf-smote-v2", ["train.py", "--cpu-only"], "predict.py"),
    ("mega-recall-blend", ["train.py"], "predict.py"),
    ("gbm-mega-blend", ["train.py"], "predict.py"),
    ("xgb-recall", ["train.py"], "predict.py"),
    ("lgb-seedblend-recall", ["train.py"], "predict.py"),
    ("knn-positive-profile", ["train.py"], "predict.py"),
    ("smote-stack-recall", ["train.py", "--cpu-only"], "predict.py"),
]

OPTIONAL_SLOW = [
    ("lightgbm-optuna", ["train.py", "--n-trials", "8"], "predict.py"),
    ("autogluon-recall", ["train.py", "--time-limit", "600"], "predict.py"),
    ("meta-recall-stack", ["train.py"], "predict.py"),
]


def run_method(method: str, train_args: list[str], predict_script: str) -> bool:
    method_dir = ROOT / "models" / method
    if not (method_dir / train_args[0]).is_file():
        print(f"SKIP {method}: missing {train_args[0]}")
        return False
    print(f"\n=== TRAIN {method} ===")
    r = subprocess.run([PYTHON, str(method_dir / train_args[0]), *train_args[1:]], cwd=ROOT)
    if r.returncode != 0:
        print(f"FAIL train {method}")
        return False
    print(f"\n=== PREDICT {method} ===")
    r = subprocess.run([PYTHON, str(method_dir / predict_script)], cwd=ROOT)
    if r.returncode != 0:
        print(f"FAIL predict {method}")
        return False
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--methods", nargs="+", default=None)
    args = parser.parse_args()

    jobs = list(CORE_TRAIN_PREDICT)
    if args.include_slow:
        jobs.extend(OPTIONAL_SLOW)

    if args.methods:
        wanted = set(args.methods)
        jobs = [j for j in jobs if j[0] in wanted]

    ok, fail = 0, 0
    for method, train_args, predict_script in jobs:
        if run_method(method, train_args, predict_script):
            ok += 1
        else:
            fail += 1
    print(f"\nDone: {ok} ok, {fail} failed")


if __name__ == "__main__":
    main()
