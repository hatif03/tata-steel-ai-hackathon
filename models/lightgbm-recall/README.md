# LightGBM Recall-First

Evolution of [`lightgbm-cv`](../lightgbm-cv/README.md) using **recall-oriented threshold tuning** instead of maximizing OOF accuracy at a high threshold.

## Method

- Same LightGBM pipeline: native NaN + missing indicators (`utils/tabular_features.py`)
- Threshold via `utils/threshold_tuning.select_recall_oriented_threshold`:
  1. 100% OOF recall if achievable without >12% positive rate
  2. Else FPR ≤ 3%
  3. Else forum fixed **t = 0.05** when recall gain vs FPR cap

## Results (latest run)

| Metric | lightgbm-recall | lightgbm-cv (old) |
|--------|-----------------|-------------------|
| Threshold strategy | fixed_t_0.05 | max_accuracy |
| Threshold | **0.05** | 0.73 |
| OOF accuracy | 92.90% | 95.86% |
| OOF recall | **42.4%** | 21.2% |
| Test positives | **19** | 5 |
| Prior LB score | — | 1.88679 |

All canary coils `{654, 806, 532, 958, 1187}` predicted `Y=1`.

## How to run

```powershell
python models/lightgbm-recall/train.py
python models/lightgbm-recall/predict.py
python scripts/check_submission_vs_baseline.py models/lightgbm-recall/submission/submission.csv
python models/lightgbm-recall/pack.py
```

**Recommended upload** for recall-first strategy (Phase 0 winner).
