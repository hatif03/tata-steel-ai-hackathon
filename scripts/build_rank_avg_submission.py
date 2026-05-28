"""Build rank-averaged ensemble submissions across multiple recall models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.threshold_tuning import apply_top_k, rank_average_proba, sweep_optimal_k  # noqa: E402
from utils.union_augment import CANARY_COILS  # noqa: E402

OUTPUT_DIR = ROOT / "models" / "phase8-rethreshold" / "outputs"

DEFAULT_METHODS = (
    "gbm-recall",
    "lightgbm-recall",
    "sklearn-recall",
    "recall-blend",
    "catboost-recall",
    "xgb-recall",
    "lgb-seedblend-recall",
    "rf-smote-v2",
    "mega-recall-blend",
)


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def load_method_proba(method: str) -> pd.DataFrame | None:
    path = ROOT / f"models/{method}/outputs/latest/predictions/test_predictions.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)[["CoilID", "proba"]].rename(columns={"proba": method})


def load_oof_for_optimal_k(methods: list[str]) -> tuple[pd.DataFrame, list[str]] | None:
    oof_col = {
        "gbm-recall": "oof_blend",
        "lightgbm-recall": "oof_proba",
        "sklearn-recall": "oof_blend",
        "recall-blend": "oof_blend",
        "catboost-recall": "oof_proba",
        "xgb-recall": "oof_proba",
        "lgb-seedblend-recall": "oof_proba",
        "rf-smote-v2": "oof_proba",
        "mega-recall-blend": "oof_blend",
    }
    merged = None
    loaded: list[str] = []
    for method in methods:
        col = oof_col.get(method, "oof_proba")
        path = ROOT / f"models/{method}/outputs/latest/oof_predictions.csv"
        if not path.is_file():
            continue
        oof = pd.read_csv(path)
        if col not in oof.columns:
            continue
        part = oof[["CoilID", "y_true", col]].rename(columns={col: method})
        merged = part if merged is None else merged.merge(part, on=["CoilID", "y_true"])
        loaded.append(method)
    if merged is None or len(loaded) < 2:
        return None
    return merged, loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--k-values", nargs="+", type=int, default=[33, 35, 38, 40])
    parser.add_argument("--oof-k-sweep", action="store_true", help="Also build OOF-optimal K variant")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    merged = None
    loaded: list[str] = []
    for method in args.methods:
        part = load_method_proba(method)
        if part is None:
            continue
        merged = part if merged is None else merged.merge(part, on="CoilID")
        loaded.append(method)

    if merged is None or len(loaded) < 2:
        raise SystemExit("Need at least 2 model prediction files")

    canary_idx = canary_indices(merged["CoilID"])
    probas = [merged[m].values.astype(float) for m in loaded]
    rank_proba = rank_average_proba(probas)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for k in args.k_values:
        name = f"rank-avg-k{k}"
        pred, eff_t = apply_top_k(rank_proba, k, force_positive_idx=canary_idx)
        out_dir = args.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"CoilID": merged["CoilID"], "Y": pred}).to_csv(
            out_dir / "submission.csv", index=False
        )
        meta = {
            "strategy": name,
            "methods": loaded,
            "k": k,
            "effective_threshold": eff_t,
            "test_positives": int(pred.sum()),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        manifest.append(meta)
        print(f"Wrote {name}: {meta['test_positives']} positives")

    if args.oof_k_sweep:
        oof_data = load_oof_for_optimal_k(loaded)
        if oof_data is not None:
            oof_merged, oof_loaded = oof_data
            oof_rank = rank_average_proba([oof_merged[m].values for m in oof_loaded])
            y = oof_merged["y_true"].values.astype(int)
            oof_canary = canary_indices(oof_merged["CoilID"])
            best_k, best_m = sweep_optimal_k(y, oof_rank, 20, 55, force_positive_idx=oof_canary)
            pred, eff_t = apply_top_k(rank_proba, best_k, force_positive_idx=canary_idx)
            name = f"rank-avg-oof-k{best_k}"
            out_dir = args.output_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"CoilID": merged["CoilID"], "Y": pred}).to_csv(
                out_dir / "submission.csv", index=False
            )
            meta = {
                "strategy": name,
                "methods": loaded,
                "oof_optimal_k": best_k,
                "oof_accuracy": best_m.accuracy,
                "effective_threshold": eff_t,
                "test_positives": int(pred.sum()),
            }
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            manifest.append(meta)
            print(f"Wrote {name}: OOF acc={best_m.accuracy:.4f}, {meta['test_positives']} test positives")

    (args.output_dir / "rank_avg_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
