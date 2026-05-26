"""Pack Phase 7 exact-K re-thresholded submissions with parent method source archives."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.submission_pack import pack_method  # noqa: E402

PHASE7_DIR = ROOT / "models" / "phase7-rethreshold" / "outputs"
METHOD_MAP = {
    "sklearn-recall": ROOT / "models" / "sklearn-recall",
    "lightgbm-recall": ROOT / "models" / "lightgbm-recall",
    "gbm-recall": ROOT / "models" / "gbm-recall",
    "recall-blend": ROOT / "models" / "recall-blend",
    "mega-recall-blend": ROOT / "models" / "mega-recall-blend",
    "lightgbm-optuna": ROOT / "models" / "lightgbm-optuna",
    "rf-smote-v2": ROOT / "models" / "rf-smote-v2",
}


def main() -> None:
    packed: list[dict] = []
    for method, method_dir in METHOD_MAP.items():
        sub_csv = PHASE7_DIR / method / "submission.csv"
        if not sub_csv.is_file():
            print(f"Skip {method}: no {sub_csv}")
            continue

        out_dir = PHASE7_DIR / method / "submission"
        out_dir.mkdir(parents=True, exist_ok=True)

        archive = pack_method(method_dir, submission_csv=sub_csv, scaffold_approach=False)
        dest_sub = out_dir / "submission.csv"
        shutil.copy2(sub_csv, dest_sub)
        dest_zip = out_dir / archive.name
        shutil.copy2(archive, dest_zip)

        meta_files = list((PHASE7_DIR / method).glob("rethreshold_k*.json"))
        meta = json.loads(meta_files[0].read_text(encoding="utf-8")) if meta_files else {}
        entry = {
            "method": method,
            "submission_csv": str(dest_sub),
            "source_archive": str(dest_zip),
            "strategy": meta.get("strategy"),
            "test_positives": meta.get("test_positives"),
        }
        packed.append(entry)
        print(f"Packed {method}: CSV={dest_sub} ZIP={dest_zip}")

    gbm33 = ROOT / "models/gbm-recall/outputs/latest/predictions/submission.csv"
    if gbm33.is_file():
        out = PHASE7_DIR / "gbm-recall-forum33" / "submission"
        out.mkdir(parents=True, exist_ok=True)
        archive = pack_method(
            METHOD_MAP["gbm-recall"],
            submission_csv=gbm33,
            scaffold_approach=False,
        )
        shutil.copy2(gbm33, out / "submission.csv")
        shutil.copy2(archive, out / archive.name)
        packed.append(
            {
                "method": "gbm-recall-forum33",
                "submission_csv": str(out / "submission.csv"),
                "source_archive": str(out / archive.name),
                "strategy": "forum_fixed_0.05",
                "test_positives": 33,
            }
        )
        print(f"Packed gbm-recall-forum33: {out / 'submission.csv'}")

    (PHASE7_DIR / "pack_manifest.json").write_text(json.dumps(packed, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
