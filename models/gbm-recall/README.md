# GBM Recall Ensemble

Equal-weight **XGBoost + LightGBM + CatBoost** with recall-first threshold (forum-aligned).

Unlike [`gbm-ensemble`](../gbm-ensemble/README.md) (failed LB 1.13208 with t=0.72), this uses **equal 1/3 blend weights** and `select_recall_oriented_threshold`.

## Results (latest run)

| Metric | Value |
|--------|-------|
| Threshold strategy | target_fpr |
| Threshold | ~0.24 |
| Test positives | 15 |
| Canary coils | all Y=1 |

## How to run

```powershell
python models/gbm-recall/train.py
python models/gbm-recall/predict.py
python models/gbm-recall/pack.py
```
