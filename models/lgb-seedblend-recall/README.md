# LGB Seed Blend Recall

Five LightGBM models (seeds 42, 123, 456, 789, 999) with **rank averaging** across OOF/test probas, then top-K=33 output.

Rank averaging preserves per-model ordering — distinct from failed probability averaging.

## How to run

```powershell
python models/lgb-seedblend-recall/train.py
python models/lgb-seedblend-recall/predict.py
python models/lgb-seedblend-recall/pack.py
```

## Results

See `outputs/latest/metrics.json` after training.
