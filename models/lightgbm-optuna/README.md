# LightGBM Optuna

**Optuna hyperparameter search** (~50 trials) optimizing **OOF PR-AUC** on 10-fold stratified CV. Same base features as lightgbm-recall. **Canary guard** rejects trials that drop test proba on coils 806/1187 below sklearn-recall baseline.

Threshold: rank-based top-K (default 26) with canaries forced positive.

## How to run

```powershell
pip install optuna
python models/lightgbm-optuna/train.py --n-trials 50 --k 26
python models/lightgbm-optuna/predict.py
python scripts/rethreshold_submission.py --methods lightgbm-optuna --write-all
python models/lightgbm-optuna/pack.py
```

## Results

| Metric | Value |
|--------|-------|
| Optuna best OOF PR-AUC | 0.345 |
| Test positives @ top_k_26 | 26 |
