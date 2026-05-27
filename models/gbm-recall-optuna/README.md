# GBM Recall Optuna

Joint Optuna tuning of XGB + LightGBM + CatBoost with equal blend. **Objective: OOF accuracy @ top-K** (Phase 9). **Rank-based canary guard** on OOF (806/1187 in top-K on train proxy). Default K=33.

## How to run

```powershell
python models/gbm-recall-optuna/train.py --n-trials 30 --k 33
python models/gbm-recall-optuna/predict.py
python scripts/rethreshold_submission.py --phase8 --methods gbm-recall-optuna --write-all --pack
```

## Results

From `outputs/latest/metrics.json` (30 trials, K=33):

| Metric | Value |
|--------|-------|
| OOF PR-AUC | 0.327 |
| OOF accuracy @ K=33 | 0.949 |
| Test positives | 33 |

Note: strict canary guard rejected all Optuna trials; final model uses study best or gbm-recall default fallback.

Prior best LB: **12.45283** (gbm-recall forum33).
