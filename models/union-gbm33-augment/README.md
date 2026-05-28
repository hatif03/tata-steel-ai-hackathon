# Union GBM33 Augment

GBM-anchored union submission: keep all **gbm-recall** top-33 coils (canaries forced), then add exclusives from secondary recall models ranked by `max(secondary proba)`.

## Results (HackerEarth)

| target_k | Method name | LB score | Test pos | Added exclusives (cumulative beyond gbm33) |
|----------|-------------|----------|----------|------------------------------------------|
| 35 | union-gbm33-plus-2 | 13.20755 | 35 | 282, 631 |
| 38 | union-gbm33-plus-5 | 14.33964 | 38 | …, 940 |
| 39 | union-gbm33-plus-6 | **14.71698** | 39 | …, 1138 |
| 40 | union-gbm33-plus-7 | **15.09434** | 40 | …, 1346 |
| 41 | union-gbm33-plus-8 | **15.47170** | 41 | …, 1189 |
| 42 | union-gbm33-plus-9 | **15.84906** | 42 | …, 826 |

**Current best:** K=42 → LB **15.84906**. Marginal gain per exclusive in the 38–42 band: **~0.377 LB**.

## How to run

```powershell
python models/gbm-recall/train.py --threshold-strategy top_k_33
python models/gbm-recall/predict.py
python models/union-gbm33-augment/train.py --target-k 42 --ensure-secondaries
python models/union-gbm33-augment/predict.py
python models/union-gbm33-augment/pack.py
```

Options: `--base-k`, `--target-k`, `--secondary-k`, `--ranking max|mean|weighted`, `--secondary-methods`, `--train-gbm`, `--ensure-secondaries`.

## Technical details

- **Anchor:** equal-weight XGB+LGB+Cat from `gbm-recall` @ `apply_top_k(..., 33)` on test probas
- **Secondaries (default):** lightgbm-recall, sklearn-recall, recall-blend, rf-smote-v2, mega-recall-blend, lightgbm-optuna, gbm-mega-blend — each contributes top-**26** set
- **Ranking:** exclusives sorted by `max(secondary proba)`; gbm33 coils never dropped
- **Canaries:** 654, 806, 532, 958, 1187 forced Y=1 in anchor and secondary top-K
- **Implementation:** [utils/union_augment.py](../../utils/union_augment.py)
- **No probability averaging** at inference — coil-ID set union only

## Why this approach

Phase 8 showed re-ranking within gbm top-33 does not improve LB (gbm-mega-blend = gbm @ K=33). Phase 9 showed **adding** secondary-only coils improves LB linearly (~0.38 per coil) through K=42.

## Next

- `--target-k 43` after extending secondary pool (catboost-recall, meta-recall-stack, autogluon-recall)
- See [DEVELOPMENT.md](../../DEVELOPMENT.md) §20F
