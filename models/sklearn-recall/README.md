# Sklearn Recall Ensemble

**RandomForest + ExtraTrees + GradientBoosting** with `class_weight={0:1, 1:30}`, median imputation in CV, equal-weight blend, recall-first threshold.

## Results (latest run)

| Metric | Value |
|--------|-------|
| Threshold strategy | target_fpr |
| Test positives | 21 |
| Canary coils | all Y=1 |

## How to run

```powershell
python models/sklearn-recall/train.py
python models/sklearn-recall/predict.py
python models/sklearn-recall/pack.py
```
