# SMOTE Stack Recall

Stacking ensemble: BorderlineSMOTE inside CV → RF + LightGBM + CatBoost bases → LogisticRegression meta → top-K=33.

Forum-aligned SMOTE path with rank-based output (not raw t=0.31).

## How to run

```powershell
python models/smote-stack-recall/train.py
python models/smote-stack-recall/predict.py
python models/smote-stack-recall/pack.py
```

RF base uses Intel Extension for Scikit-learn when available (`--cpu-only` to disable).

## Results

See `outputs/latest/metrics.json` after training.
