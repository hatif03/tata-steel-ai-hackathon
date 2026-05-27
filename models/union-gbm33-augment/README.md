# Union GBM33 Augment

GBM-anchored union submission: keep all **gbm-recall** top-33 coils (canaries forced), then add exclusives from secondary recall models ranked by `max(secondary proba)`.

## Results

| Metric | Value |
|--------|-------|
| Prior best LB | 14.33964 (phase8 union-gbm33-plus-5) |
| Default target_k | 38 |

## How to run

```powershell
python models/gbm-recall/train.py --threshold-strategy top_k_33
python models/gbm-recall/predict.py
python models/union-gbm33-augment/train.py --target-k 38 --ensure-secondaries
python models/union-gbm33-augment/predict.py
python models/union-gbm33-augment/pack.py
```

Options: `--base-k`, `--target-k`, `--secondary-k`, `--ranking max|mean|weighted`, `--secondary-methods`, `--train-gbm`, `--ensure-secondaries`.

## Technical details

- Anchor: equal-weight XGB+LGB+Cat from `gbm-recall` @ top-K=33
- Secondaries: lightgbm-recall, sklearn-recall, recall-blend, rf-smote-v2, mega-recall-blend, lightgbm-optuna, gbm-mega-blend (skip if outputs missing)
- Union logic: [utils/union_augment.py](../../utils/union_augment.py)
- No probability averaging at inference — coil-ID union only
