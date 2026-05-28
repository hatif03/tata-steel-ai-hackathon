# CatBoost Recall — seed 42 (depth 6 variant)

CatBoost with `random_seed=42`, **depth=6**, iterations=1000 for vote-union pool diversity. Distinct from base `catboost-recall` (depth 5, 800 iters).

## How to run

```powershell
python models/catboost-recall-seed42/train.py --data-dir dataset
python models/catboost-recall-seed42/predict.py --data-dir dataset
```

## Results

See `outputs/latest/metrics.json` after training.
