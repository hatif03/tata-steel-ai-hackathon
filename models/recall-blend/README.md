# Recall Blend (Sklearn + LightGBM)

**50/50 probability blend** of sklearn-recall (RF+ET+GBM, class_weight 1:30, median impute) and lightgbm-recall (same tabular features, scale_pos_weight). Threshold tuned for ~7.7% OOF positive rate (forum ~26/339 pattern).

## Why this approach

sklearn-recall (LB 7.92) and lightgbm-recall (LB 7.17) make different errors; averaging OOF probabilities is low cost and aligns with forum ensemble winners. Phase 6 targets **24–28 test positives** with all canary coils positive.

## Technical details

- Features: `utils/tabular_features.py` (X1–X49 + missing indicators + row_missing_count)
- 5-fold stratified CV; median impute inside folds for sklearn branch only
- Blend: `0.5 * sklearn_trio + 0.5 * lgbm`
- Default threshold: `target_rate` (OOF ~7.7% positives); re-sweep with `scripts/rethreshold_submission.py` after predict

## How to run

```powershell
python models/recall-blend/train.py
python models/recall-blend/predict.py
python scripts/check_submission_vs_baseline.py models/recall-blend/outputs/latest/predictions/submission.csv
python models/recall-blend/pack.py
```

## Results

| Metric | Value |
|--------|-------|
| OOF PR-AUC | 0.363 |
| OOF accuracy @ threshold | 0.923 |
| Threshold (rethreshold) | 0.2208 |
| Test positives (rethreshold) | 24 |
| Test positives (train default) | 27 |

Update from `outputs/latest/metrics.json` after each run.
