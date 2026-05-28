# GBM Recall Optuna

Joint Optuna tuning of XGB + LightGBM + CatBoost with equal blend. **Objective: OOF accuracy @ top-K** (Phase 9). **Rank-based canary guard** on OOF (806/1187 in top-K on train proxy). Default K=33.

## How to run

```powershell
python models/gbm-recall-optuna/train.py --n-trials 30 --k 33
python models/gbm-recall-optuna/predict.py
python scripts/rethreshold_submission.py --phase8 --methods gbm-recall-optuna --write-all --pack
```

## Results

| Prior best LB | 12.45283 (gbm-recall forum33) |
| Phase 9 best (8 trials) | OOF PR-AUC 0.361, OOF acc @ K=33 |
| Test positives @ K=33 | 33 |
