# Union GBM33 Augment

GBM-anchored union submission: keep all **gbm-recall** top-33 coils (canaries forced), then add exclusives from secondary recall models ranked by `max(secondary proba)`.

## Results (HackerEarth)

| target_k | Method name | LB score | Test pos | Marginal exclusive |
|----------|-------------|----------|----------|-------------------|
| 35 | union-gbm33-plus-2 | 13.20755 | 35 | 282, 631 |
| 38 | union-gbm33-plus-5 | 14.33964 | 38 | … |
| 42 | union-gbm33-plus-9 | 15.84906 | 42 | 826 |
| 43 | union-gbm33-plus-10 | **16.22642** | 43 | 1023 |
| 44 | union-gbm33-plus-11 | **16.60377** | 44 | 804 |
| 45 | union-gbm33-plus-12 | **16.98113** | 45 | 302 |
| 46 | union-gbm33-plus-13 | **17.35849** | 46 | 582 |
| 47 | union-gbm33-plus-14 | **17.73585** | 47 | 972 |
| 48 | union-gbm33-plus-15 | **18.11321** | 48 | 867 |
| 49 | union-gbm33-plus-16 | **18.49057** | 49 | 176 |
| 50 | union-gbm33-plus-17 | **18.49057** | 50 | 797 (no gain) |

**Current best:** K=49 → LB **18.49057** (`union-gbm33-plus-16`). Marginal gain per exclusive in K=38–49 band: **~0.377 LB** (constant).

**Saturation:** Coil 797 @ K=50 — same LB as K=49.

## How to run

```powershell
python models/gbm-recall/train.py --threshold-strategy top_k_33
python models/gbm-recall/predict.py
python scripts/run_phase10_train_batch.py --include-slow
python models/union-gbm33-augment/train.py --target-k 49 --ensure-secondaries
python models/union-gbm33-augment/predict.py
python models/union-gbm33-augment/pack.py
```

## Technical details

- **Anchor:** equal-weight XGB+LGB+Cat from `gbm-recall` @ `apply_top_k(..., 33)` on test probas
- **Secondaries (16 models):** lightgbm-recall, sklearn-recall, recall-blend, rf-smote-v2, mega-recall-blend, lightgbm-optuna, gbm-mega-blend, catboost-recall, meta-recall-stack, autogluon-recall, xgb-recall, lgb-seedblend-recall, knn-positive-profile, smote-stack-recall — each top-**26**
- **Ranking:** exclusives sorted by `max(secondary proba)`; gbm33 coils never dropped
- **Canaries:** 654, 806, 532, 958, 1187 forced Y=1
- **Implementation:** [utils/union_augment.py](../../utils/union_augment.py)

## What failed on LB (Phase 10)

| Method | LB | Verdict |
|--------|-----|---------|
| rank-avg-k35 | 13.20755 | Same as plus-2 — rank avg hurts |
| vote-union-m2-k33 | 17.73585 | Same as plus-14 — no gain vs union |

## Next

- Rebuild union after Colab AutoGluon `best_quality` for fresh exclusives beyond 797
- See [DEVELOPMENT.md](../../DEVELOPMENT.md) §21F
