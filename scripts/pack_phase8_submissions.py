"""Pack Phase 8 GBM-anchored re-threshold and union submissions."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.submission_pack import pack_method  # noqa: E402

PHASE8_DIR = ROOT / "models" / "phase8-rethreshold" / "outputs"
METHOD_MAP = {
    "gbm-recall": ROOT / "models" / "gbm-recall",
    "gbm-mega-blend": ROOT / "models" / "gbm-mega-blend",
    "gbm-recall-optuna": ROOT / "models" / "gbm-recall-optuna",
}


def pack_one(name: str, sub_csv: Path, method_dir: Path) -> dict | None:
    if not sub_csv.is_file():
        return None
    out_dir = PHASE8_DIR / name / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = pack_method(method_dir, submission_csv=sub_csv, scaffold_approach=False)
    dest_sub = out_dir / "submission.csv"
    shutil.copy2(sub_csv, dest_sub)
    dest_zip = out_dir / archive.name
    shutil.copy2(archive, dest_zip)
    return {
        "name": name,
        "submission_csv": str(dest_sub),
        "source_archive": str(dest_zip),
    }


def main() -> None:
    packed: list[dict] = []

    for method in ("gbm-recall", "gbm-mega-blend", "gbm-recall-optuna"):
        sub = PHASE8_DIR / method / "submission.csv"
        if method not in METHOD_MAP:
            continue
        entry = pack_one(method, sub, METHOD_MAP[method])
        if entry:
            meta_files = list((PHASE8_DIR / method).glob("rethreshold_k*.json"))
            if meta_files:
                entry.update(json.loads(meta_files[0].read_text(encoding="utf-8")))
            packed.append(entry)
            print(f"Packed {method}: {entry['submission_csv']}")

    for union_dir in PHASE8_DIR.glob("union-gbm33-plus-*"):
        sub = union_dir / "submission.csv"
        entry = pack_one(union_dir.name, sub, METHOD_MAP["gbm-recall"])
        if entry:
            meta_path = union_dir / "meta.json"
            if meta_path.is_file():
                entry.update(json.loads(meta_path.read_text(encoding="utf-8")))
            packed.append(entry)
            print(f"Packed {union_dir.name}")

    (PHASE8_DIR / "pack_manifest.json").write_text(json.dumps(packed, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
