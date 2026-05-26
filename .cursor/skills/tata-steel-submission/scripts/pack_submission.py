"""Build HackerEarth submission folder and zip/tar archive for a method."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_artifacts import repo_root  # noqa: E402
from utils.submission_pack import pack_method, scaffold_approach_txt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package HackerEarth prediction CSV + approach.txt + source into submission/"
    )
    parser.add_argument(
        "method",
        type=Path,
        help="Method folder (e.g. models/xgboost-baseline)",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Use a specific run instead of latest")
    parser.add_argument(
        "--submission-csv",
        type=Path,
        default=None,
        help="Use this submission.csv instead of latest run predictions",
    )
    parser.add_argument(
        "--format",
        choices=("zip", "tar.gz"),
        default="zip",
        help="Archive format for source bundle (default: zip)",
    )
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument(
        "--scaffold-approach",
        action="store_true",
        help="Create approach.txt from template if missing (won't overwrite existing)",
    )
    args = parser.parse_args()

    method_dir = args.method
    if not method_dir.is_absolute():
        method_dir = repo_root() / method_dir

    if args.scaffold_approach:
        path = scaffold_approach_txt(method_dir)
        print(f"Approach file: {path}")

    archive = pack_method(
        method_dir,
        run_dir=args.run_dir,
        submission_csv=args.submission_csv,
        archive_format=args.format,
        skip_validate=args.skip_validate,
        scaffold_approach=False,
    )
    submission_dir = method_dir / "submission"
    print(f"Submission CSV: {submission_dir / 'submission.csv'}")
    print(f"Approach text:  {submission_dir / 'approach.txt'}")
    print(f"Source copies:  {submission_dir / 'source'}")
    print(f"Archive:        {archive}")
    print("")
    print("Upload to HackerEarth (two separate files):")
    print(f"  1. Predictions: {submission_dir / 'submission.csv'}")
    print(f"  2. Source code: {archive}")


if __name__ == "__main__":
    main()
