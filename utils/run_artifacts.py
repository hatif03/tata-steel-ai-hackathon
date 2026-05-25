"""Standard run output directories for Tata Steel hackathon experiments."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_SUBDIRS = ("artifacts", "plots", "predictions")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def method_outputs_root(method_dir: Path) -> Path:
    return method_dir / "outputs"


def create_run_dir(method_dir: Path, run_id: str | None = None) -> Path:
    """Create a timestamped run folder and update outputs/latest_run.txt."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = method_outputs_root(method_dir) / "runs" / run_id
    for name in RUN_SUBDIRS:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    set_latest_run(method_dir, run_id)
    return run_dir


def set_latest_run(method_dir: Path, run_id: str) -> None:
    outputs = method_outputs_root(method_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "latest_run.txt").write_text(run_id, encoding="utf-8")


def get_latest_run_dir(method_dir: Path) -> Path:
    latest_file = method_outputs_root(method_dir) / "latest_run.txt"
    if not latest_file.is_file():
        raise FileNotFoundError(
            f"No latest run for {method_dir.name}. Train first: "
            f"python models/{method_dir.name}/train.py"
        )
    run_id = latest_file.read_text(encoding="utf-8").strip()
    run_dir = method_outputs_root(method_dir) / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Latest run directory missing: {run_dir}")
    return run_dir


def resolve_run_dir(method_dir: Path, run_dir: Path | None = None) -> Path:
    return run_dir if run_dir is not None else get_latest_run_dir(method_dir)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_metrics(run_dir: Path, metrics: dict[str, Any]) -> Path:
    path = run_dir / "metrics.json"
    save_json(path, metrics)
    return path


def save_run_config(run_dir: Path, config: dict[str, Any]) -> Path:
    path = run_dir / "run_config.json"
    save_json(path, config)
    return path


def copy_to_latest_summary(method_dir: Path, run_dir: Path) -> None:
    """Copy metrics, plots, predictions, and model artifacts to outputs/latest/."""
    summary = method_outputs_root(method_dir) / "latest"
    if summary.exists():
        shutil.rmtree(summary)
    summary.mkdir(parents=True)

    for rel in ("metrics.json", "run_config.json", "oof_predictions.csv"):
        src = run_dir / rel
        if src.is_file():
            shutil.copy2(src, summary / rel)

    plots_src = run_dir / "plots"
    if plots_src.is_dir():
        shutil.copytree(plots_src, summary / "plots")

    artifacts_src = run_dir / "artifacts"
    if artifacts_src.is_dir():
        shutil.copytree(artifacts_src, summary / "artifacts")

    predictions_src = run_dir / "predictions"
    if predictions_src.is_dir():
        shutil.copytree(predictions_src, summary / "predictions")


def write_artifacts_manifest(artifacts_dir: Path) -> Path:
    """Write manifest.json listing every saved model/preprocessor file."""
    manifest: dict[str, Any] = {"files": {}}
    for path in sorted(artifacts_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][path.name] = {
                "size_bytes": path.stat().st_size,
                "path": str(path.resolve()),
            }
    manifest_path = artifacts_dir / "manifest.json"
    save_json(manifest_path, manifest)
    return manifest_path


def print_saved_artifacts(artifacts_dir: Path) -> None:
    manifest_path = artifacts_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = load_json(manifest_path)
    print("Saved model artifacts:")
    for name, info in manifest.get("files", {}).items():
        size_kb = info["size_bytes"] / 1024
        print(f"  - {name} ({size_kb:.1f} KB)")
