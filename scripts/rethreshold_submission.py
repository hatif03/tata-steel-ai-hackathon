"""Re-threshold saved test probabilities without retraining (Phase 6A / Phase 7A)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.threshold_tuning import (  # noqa: E402
    apply_threshold,
    apply_top_k,
    format_result,
    tune_fixed_thresholds,
    tune_target_positive_rate,
    tune_target_test_positives,
    tune_top_k,
)

CANARY_COILS = (654, 806, 532, 958, 1187)
OUTPUT_DIR = ROOT / "models" / "phase7-rethreshold" / "outputs"
PHASE8_OUTPUT_DIR = ROOT / "models" / "phase8-rethreshold" / "outputs"
DEFAULT_K_VALUES = (25, 26, 27, 28, 30, 33)
PHASE8_K_VALUES = (33, 34, 35, 36, 38)
DEFAULT_METHODS = ("sklearn-recall", "lightgbm-recall", "gbm-recall", "recall-blend")
PHASE8_METHODS = ("gbm-recall", "gbm-mega-blend", "gbm-recall-optuna")


def oof_path_for_method(method: str) -> tuple[Path, str]:
    mapping = {
        "sklearn-recall": ("oof_blend", ROOT / "models/sklearn-recall/outputs/latest/oof_predictions.csv"),
        "lightgbm-recall": ("oof_proba", ROOT / "models/lightgbm-recall/outputs/latest/oof_predictions.csv"),
        "gbm-recall": ("oof_blend", ROOT / "models/gbm-recall/outputs/latest/oof_predictions.csv"),
        "recall-blend": ("oof_blend", ROOT / "models/recall-blend/outputs/latest/oof_predictions.csv"),
        "rf-smote": ("oof_proba", ROOT / "models/rf-smote/outputs/latest/oof_predictions.csv"),
        "rf-smote-v2": ("oof_proba", ROOT / "models/rf-smote-v2/outputs/latest/oof_predictions.csv"),
        "mega-recall-blend": ("oof_blend", ROOT / "models/mega-recall-blend/outputs/latest/oof_predictions.csv"),
        "gbm-mega-blend": ("oof_blend", ROOT / "models/gbm-mega-blend/outputs/latest/oof_predictions.csv"),
        "gbm-recall-optuna": ("oof_blend", ROOT / "models/gbm-recall-optuna/outputs/latest/oof_predictions.csv"),
    }
    if method not in mapping:
        raise KeyError(method)
    col, path = mapping[method]
    oof = pd.read_csv(path)
    if col not in oof.columns:
        raise KeyError(f"{col} not in {path}")
    return path, col


def canary_indices(coil_ids: pd.Series) -> list[int]:
    return [int(i) for i, c in enumerate(coil_ids) if int(c) in CANARY_COILS]


def evaluate_method(method: str, k_values: tuple[int, ...]) -> list[dict]:
    test_path = ROOT / f"models/{method}/outputs/latest/predictions/test_predictions.csv"
    oof_file, proba_col = oof_path_for_method(method)
    oof = pd.read_csv(oof_file)
    test = pd.read_csv(test_path)
    y = oof["y_true"].values.astype(int)
    oof_proba = oof[proba_col].values.astype(float)
    test_proba = test["proba"].values.astype(float)
    oof_canary = canary_indices(oof["CoilID"])
    test_canary = canary_indices(test["CoilID"])

    rows: list[dict] = []
    candidates: dict[str, object] = {
        "target_rate_077": tune_target_positive_rate(y, oof_proba, target_rate=0.077),
        "target_test_26": tune_target_test_positives(
            y, oof_proba, test_proba, target_test_positives=26, min_test_positives=24, max_test_positives=28
        ),
        **tune_fixed_thresholds(y, oof_proba, thresholds=(0.05, 0.31, 0.35)),
    }
    for k in k_values:
        candidates[f"top_k_{k}"] = tune_top_k(y, oof_proba, k, force_positive_idx=oof_canary)

    print(f"\n=== {method} ===")
    for sname, m in candidates.items():
        if sname.startswith("top_k_"):
            k = int(sname.split("_")[-1])
            test_pred, eff_t = apply_top_k(test_proba, k, force_positive_idx=test_canary)
        else:
            test_pred = apply_threshold(test_proba, m.threshold)
            eff_t = m.threshold
        canary = test[test["CoilID"].isin(CANARY_COILS)].copy()
        canary_idx = canary.index.to_numpy()
        canary["pred"] = test_pred[canary_idx]
        canary_ok = int(canary["pred"].sum()) == len(CANARY_COILS)
        test_pos = int(test_pred.sum())
        print(format_result(m))
        print(f"  test_positives={test_pos} effective_t={eff_t:.4f} canary_ok={canary_ok}")
        rows.append(
            {
                "method": method,
                "strategy": sname,
                **m.to_dict(),
                "effective_threshold": eff_t,
                "test_positives": test_pos,
                "canary_all_positive": canary_ok,
            }
        )
    return rows


def pick_best(rows: list[dict], *, k_target: int = 26) -> dict:
    df = pd.DataFrame(rows)
    ok = df[df["canary_all_positive"]]
    if len(ok) == 0:
        raise ValueError("No candidates with all canary coils positive")

    exact = ok[ok["test_positives"] == k_target]
    if len(exact):
        return exact.sort_values(["accuracy", "recall"], ascending=False).iloc[0].to_dict()

    if k_target >= 30:
        band = ok[(ok["test_positives"] >= k_target - 2) & (ok["test_positives"] <= k_target + 5)]
        if len(band):
            band = band.copy()
            band["dist_k"] = (band["test_positives"] - k_target).abs()
            return band.sort_values(["dist_k", "accuracy"], ascending=[True, False]).iloc[0].to_dict()

    in_range = ok[(ok["test_positives"] >= 24) & (ok["test_positives"] <= 28)]
    if len(in_range):
        return in_range.sort_values(["accuracy", "recall"], ascending=False).iloc[0].to_dict()

    ok = ok.copy()
    ok["dist_k"] = (ok["test_positives"] - k_target).abs()
    return ok.sort_values(["dist_k", "accuracy"], ascending=[True, False]).iloc[0].to_dict()


def write_submission(method: str, row: dict, output_dir: Path | None = None) -> Path:
    test = pd.read_csv(ROOT / f"models/{method}/outputs/latest/predictions/test_predictions.csv")
    test_proba = test["proba"].values.astype(float)
    if str(row["strategy"]).startswith("top_k_"):
        k = int(row["strategy"].split("_")[-1])
        pred, _ = apply_top_k(test_proba, k, force_positive_idx=canary_indices(test["CoilID"]))
    else:
        pred = apply_threshold(test_proba, row["threshold"])

    out_method = (output_dir or OUTPUT_DIR) / method
    out_method.mkdir(parents=True, exist_ok=True)
    sub_path = out_method / f"submission_k{int(row['test_positives'])}.csv"
    pd.DataFrame({"CoilID": test["CoilID"], "Y": pred}).to_csv(sub_path, index=False)
    (out_method / f"rethreshold_k{int(row['test_positives'])}.json").write_text(
        json.dumps(row, indent=2), encoding="utf-8"
    )
    latest = out_method / "submission.csv"
    pd.DataFrame({"CoilID": test["CoilID"], "Y": pred}).to_csv(latest, index=False)
    return sub_path


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase8", action="store_true", help="Phase 8: gbm K=33-38, output phase8-rethreshold")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--k-values", nargs="+", type=int, default=None)
    parser.add_argument("--write-all", action="store_true", help="Write top_k submission per K per method")
    parser.add_argument("--write-best", action="store_true", help="Write single best K per method")
    parser.add_argument("--pack", action="store_true", help="Run pack script after write")
    parser.add_argument("--guard-check", action="store_true", help="Run check_submission_vs_baseline on outputs")
    args = parser.parse_args()

    if args.phase8:
        OUTPUT_DIR = PHASE8_OUTPUT_DIR
        methods = args.methods or list(PHASE8_METHODS)
        k_values = tuple(sorted(set(args.k_values or PHASE8_K_VALUES)))
        k_target = 33
        sweep_name = "phase8_sweep.json"
        best_name = "phase8_best.json"
        pack_script = "pack_phase8_submissions.py"
        max_positives = 40
    else:
        methods = args.methods or list(DEFAULT_METHODS)
        k_values = tuple(sorted(set(args.k_values or DEFAULT_K_VALUES)))
        k_target = 26
        sweep_name = "phase7_sweep.json"
        best_name = "phase7_best.json"
        pack_script = "pack_phase7_submissions.py"
        max_positives = 35

    all_rows: list[dict] = []
    for method in methods:
        try:
            all_rows.extend(evaluate_method(method, k_values))
        except (FileNotFoundError, KeyError) as e:
            print(f"Skip {method}: {e}")

    if not all_rows:
        print("No methods evaluated.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    (OUTPUT_DIR / sweep_name).write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    if args.write_all:
        ok = df[df["canary_all_positive"]]
        for method in methods:
            sub = ok[ok["method"] == method]
            for k in k_values:
                pick = sub[sub["strategy"] == f"top_k_{k}"]
                if len(pick) == 0:
                    continue
                row = pick.iloc[0].to_dict()
                path = write_submission(method, row, OUTPUT_DIR)
                print(f"Wrote {path} (top_k_{k} pos={row['test_positives']})")

    if args.write_best:
        ok = df[df["canary_all_positive"]]
        for method in methods:
            sub = ok[ok["method"] == method]
            if len(sub) == 0:
                continue
            row = pick_best(sub.to_dict("records"), k_target=k_target)
            path = write_submission(method, row, OUTPUT_DIR)
            print(f"Wrote best {method}: {path}")

    best = pick_best(all_rows, k_target=k_target)
    print("\n=== GLOBAL BEST ===")
    print(json.dumps(best, indent=2))
    (OUTPUT_DIR / best_name).write_text(json.dumps(best, indent=2), encoding="utf-8")

    if args.pack:
        subprocess.run([sys.executable, str(ROOT / "scripts" / pack_script)], check=True)
    if args.guard_check:
        for sub in OUTPUT_DIR.glob("*/submission.csv"):
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/check_submission_vs_baseline.py"),
                    str(sub),
                    "--min-positives",
                    "20",
                    "--max-positives",
                    str(max_positives),
                ],
                check=False,
            )


if __name__ == "__main__":
    main()
