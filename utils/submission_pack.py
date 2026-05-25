"""Build HackerEarth submission folder and zip/tar archive for a method."""

from __future__ import annotations

import json
import shutil
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from utils.run_artifacts import get_latest_run_dir, repo_root, save_json

SOURCE_FILES = ("README.md", "train.py", "predict.py", "features.py")


def validate_submission_csv(submission_path: Path, test_path: Path) -> list[str]:
    """Validate submission CSV; returns list of error strings (empty if OK)."""
    import pandas as pd

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
        errors.append("Duplicate CoilID values in submission")
    missing = set(test["CoilID"]) - set(sub["CoilID"])
    extra = set(sub["CoilID"]) - set(test["CoilID"])
    if missing:
        errors.append(f"Missing test CoilIDs ({len(missing)})")
    if extra:
        errors.append(f"Extra CoilIDs not in test ({len(extra)})")
    if "Y" in sub.columns:
        invalid = sub[~sub["Y"].isin([0, 1])]
        if len(invalid):
            errors.append(f"Y must be 0 or 1; {len(invalid)} invalid rows")
    return errors


def copy_source_files(method_dir: Path, source_dir: Path) -> list[Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []

    notebook = method_dir / f"{method_dir.name}.ipynb"
    if notebook.is_file():
        dest = source_dir / notebook.name
        shutil.copy2(notebook, dest)
        copied.append(dest)

    for name in SOURCE_FILES:
        src = method_dir / name
        if src.is_file():
            dest = source_dir / name
            shutil.copy2(src, dest)
            copied.append(dest)

    req = repo_root() / "requirements.txt"
    if req.is_file():
        dest = source_dir / "requirements.txt"
        shutil.copy2(req, dest)
        copied.append(dest)

    utils_dir = source_dir / "utils"
    utils_dir.mkdir(exist_ok=True)
    for util_name in (
        "plotting.py",
        "run_artifacts.py",
        "submission_pack.py",
        "tabular_features.py",
        "tabular_features_enriched.py",
    ):
        util = repo_root() / "utils" / util_name
        if util.is_file():
            dest = utils_dir / util_name
            shutil.copy2(util, dest)
            copied.append(dest)

    return copied


def write_archive(archive_path: Path, approach_path: Path, source_dir: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(approach_path, arcname="approach.txt")
            for path in sorted(source_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=str(Path("source") / path.relative_to(source_dir)))
    elif archive_path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(approach_path, arcname="approach.txt")
            for path in sorted(source_dir.rglob("*")):
                if path.is_file():
                    tf.add(path, arcname=str(Path("source") / path.relative_to(source_dir)))
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def scaffold_approach_txt(method_dir: Path, *, overwrite: bool = False) -> Path:
    """Create submission/approach.txt from template if missing."""
    submission_root = method_dir / "submission"
    submission_root.mkdir(parents=True, exist_ok=True)
    approach_path = submission_root / "approach.txt"
    if approach_path.is_file() and not overwrite:
        return approach_path

    template_path = repo_root() / "utils" / "templates" / "approach.txt"
    if template_path.is_file():
        text = template_path.read_text(encoding="utf-8")
        text = text.replace("{METHOD_NAME}", method_dir.name)
        text = text.replace(
            "{PROBLEM_URL}",
            "https://www.hackerearth.com/challenges/competitive/tata-steel-ai-hackathon/machine-learning/fd-a5a6dcb2/",
        )
    else:
        text = (
            f"Tata Steel AI Hackathon — {method_dir.name}\n\n"
            "1. APPROACH\n(Describe model, validation, threshold strategy)\n\n"
            "2. FEATURE ENGINEERING\n(Describe features used/excluded, missing values)\n\n"
            "3. TOOLS\n(Python, libraries, environment)\n\n"
            "4. REPRODUCTION\n(Commands to regenerate submission.csv)\n"
        )
    approach_path.write_text(text, encoding="utf-8")
    return approach_path


def pack_method(
    method_dir: Path,
    *,
    run_dir: Path | None = None,
    archive_format: Literal["zip", "tar.gz"] = "zip",
    skip_validate: bool = False,
    scaffold_approach: bool = True,
) -> Path:
    """Copy predictions, bundle source + approach.txt, build upload archive."""
    method_dir = method_dir.resolve()
    if not method_dir.is_dir():
        raise FileNotFoundError(f"Method folder not found: {method_dir}")

    if scaffold_approach:
        scaffold_approach_txt(method_dir)

    run_dir = run_dir or get_latest_run_dir(method_dir)
    run_submission = run_dir / "predictions" / "submission.csv"
    if not run_submission.is_file():
        raise FileNotFoundError(
            f"No submission.csv in {run_dir / 'predictions'}. "
            f"Run: python models/{method_dir.name}/predict.py"
        )

    submission_root = method_dir / "submission"
    submission_root.mkdir(parents=True, exist_ok=True)
    approach_path = submission_root / "approach.txt"
    if not approach_path.is_file():
        raise FileNotFoundError(
            f"Missing {approach_path}. Fill in approach, feature engineering, and tools."
        )

    dest_submission = submission_root / "submission.csv"
    shutil.copy2(run_submission, dest_submission)

    test_path = repo_root() / "dataset" / "test.csv"
    if not skip_validate:
        errors = validate_submission_csv(dest_submission, test_path)
        if errors:
            raise SystemExit("Submission validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    source_dir = submission_root / "source"
    if source_dir.exists():
        shutil.rmtree(source_dir)
    copy_source_files(method_dir, source_dir)

    ext = ".zip" if archive_format == "zip" else ".tar.gz"
    archive_path = submission_root / f"{method_dir.name}-hackerearth{ext}"
    if archive_path.is_file():
        archive_path.unlink()
    write_archive(archive_path, approach_path, source_dir)

    save_json(
        submission_root / "pack_meta.json",
        {
            "method": method_dir.name,
            "packed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir.resolve()),
            "submission_csv": str(dest_submission.resolve()),
            "archive": str(archive_path.resolve()),
            "approach_txt": str(approach_path.resolve()),
            "upload_instructions": {
                "prediction_file": str(dest_submission.resolve()),
                "source_archive": str(archive_path.resolve()),
                "problem_url": "https://www.hackerearth.com/challenges/competitive/tata-steel-ai-hackathon/machine-learning/fd-a5a6dcb2/",
            },
        },
    )
    return archive_path
