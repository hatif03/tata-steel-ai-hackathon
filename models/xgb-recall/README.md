# XGB Recall

Solo XGBoost classifier with stratified 5-fold CV and **top-K=33** recall-first output (canary coils forced positive).

## Method

- Features: raw X1–X49 via `utils/tabular_features`
- `XGBClassifier`: 800 trees, depth 5, lr 0.03, `scale_pos_weight ≈ 19.5`
- Threshold: rank-based top-K (not fixed t=0.05)

## How to run

```powershell
python models/xgb-recall/train.py
python models/xgb-recall/predict.py
python models/xgb-recall/pack.py
```

## Results

See `outputs/latest/metrics.json` after training.
