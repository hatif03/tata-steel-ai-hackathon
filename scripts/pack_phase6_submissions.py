"""Pack Phase 6 re-thresholded submissions with parent method source archives."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.submission_pack import pack_method  # noqa: E402

PHASE6_DIR = ROOT / "models" / "phase6-rethreshold" / "outputs"
METHOD_MAP = {
    "sklearn-recall": ROOT / "models" / "sklearn-recall",
    "lightgbm-recall": ROOT / "models" / "lightgbm-recall",
    "gbm-recall": ROOT / "models" / "gbm-recall",
    "recall-blend": ROOT / "models" / "recall-blend",
}


def main() -> None:
    packed: list[dict] = []
    for method, method_dir in METHOD_MAP.items():
        sub_csv = PHASE6_DIR / method / "submission.csv"
        meta_path = PHASE6_DIR / method / "rethreshold_meta.json"
        if not sub_csv.is_file():
            print(f"Skip {method}: no {sub_csv}")
            continue

        out_dir = PHASE6_DIR / method / "submission"
        out_dir.mkdir(parents=True, exist_ok=True)

        archive = pack_method(
            method_dir,
            submission_csv=sub_csv,
            scaffold_approach=False,
        )

        # Copy zip + submission into phase6 folder for easy upload
        import shutil

        dest_sub = out_dir / "submission.csv"
        shutil.copy2(sub_csv, dest_sub)
        dest_zip = out_dir / archive.name
        shutil.copy2(archive, dest_zip)

        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        entry = {
            "method": method,
            "submission_csv": str(dest_sub),
            "source_archive": str(dest_zip),
            "threshold": meta.get("threshold"),
            "test_positives": meta.get("test_positives"),
        }
        packed.append(entry)
        print(f"Packed {method}:")
        print(f"  CSV: {dest_sub}")
        print(f"  ZIP: {dest_zip}")

    (PHASE6_DIR / "pack_manifest.json").write_text(json.dumps(packed, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {PHASE6_DIR / 'pack_manifest.json'}")


if __name__ == "__main__":
    main()
