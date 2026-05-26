# GBM Mega Blend

GBM-recall primary (70%) + sklearn + lightgbm secondary at **K=33**. Strategies: gbm_only, gbm_heavy, weight-opt, stacking.

## How to run

```powershell
python models/gbm-mega-blend/train.py --k 33
python models/gbm-mega-blend/predict.py
python scripts/rethreshold_submission.py --phase8 --methods gbm-mega-blend --write-best --pack
```

## Results

From `outputs/latest/metrics.json` (K=33, stacking selected):

| Metric | Value |
|--------|-------|
| OOF accuracy @ K=33 | 0.953 |
| OOF PR-AUC | 0.360 |
| Test positives | 33 |

**LB score:** **12.45283** @ 33 positives — identical to gbm-recall at K=33 (same top-33 coil set; stacking changed probas but not who gets flagged).

Current project best: **14.33964** (union-gbm33-plus-5). See [`models/phase8-rethreshold/README.md`](../phase8-rethreshold/README.md).
