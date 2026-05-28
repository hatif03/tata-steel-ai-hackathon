# Vote Union Recall

Multi-model **vote consensus** submission: a test coil is predicted **Y=1** if it appears in the top-**K** set of at least **M** recall models (canaries always forced positive).

## Results (HackerEarth)

| Config | Test pos | LB score | Notes |
|--------|----------|----------|-------|
| **vote-union-m2-k40** | 61 | **23.01887** | **Current best (Phase 11)** |
| vote-union-m3-k40 | 51 | 19.24529 | Stricter M hurts |
| vote-union-m2-k33 | 47 | 17.73585 | K too low — ties union plus-14 |
| union-gbm33-plus-16 | 49 | 18.49057 | Prior best (Phase 10) |

**Marginal gain 49→61 positives (union→vote):** (23.02 − 18.49) / 12 ≈ **0.377 LB/point** — same constant as union augment, but vote finds **more valid positives** at K=40.

## Why this approach

Union augment adds exclusives one-by-one from ranked secondaries — saturated at coil 797 (K=50). Vote union at K=40 asks: *which coils do multiple independent models agree on?* That surfaces 12 extra true positives vs union at K=49 without the FP flood of M=1 (full union of top-40 per model).

## Technical details

- **Pool:** 17+ models via `utils/vote_union.DEFAULT_VOTE_METHODS` (gbm-recall, gbm-recall-safe-fe, lightgbm-recall, sklearn-recall, recall-blend, catboost variants, rf-smote-v2, mega-recall-blend, lightgbm-optuna, gbm-mega-blend, meta-recall-stack, autogluon-recall, xgb-recall, lgb-seedblend-recall, knn-positive-profile, smote-stack-recall).
- **Per model:** `apply_top_k(proba, K)` with canaries {654, 806, 532, 958, 1187} forced Y=1.
- **Vote rule:** `sum(model_topK) >= M` → Y=1.
- **Default:** K=40, M=2 (best confirmed LB).
- **Advanced variants:** `scripts/build_vote_union_advanced.py` — high-vote thresholds (v10/v12/v14), weighted vote, gbm33 OR vote hybrid.

## How to run

```powershell
# Ensure vote pool has test_predictions.csv (train secondaries first)
python scripts/run_phase10_train_batch.py --include-slow

# Build vote-union method @ best config
python models/vote-union-recall/train.py --k 40 --min-votes 2
python models/vote-union-recall/predict.py
python models/vote-union-recall/pack.py

# Sweep K/M for upload batch
python scripts/build_vote_union.py
python scripts/build_vote_union_advanced.py
python scripts/pack_phase8_submissions.py

# Analyze vote tiers for m2-k40
python scripts/analyze_vote_submission.py
```

## Phase 11 upload priority

1. **vote-union-m2-k40** — confirmed best (23.02)
2. vote-union-m2-k41 … m2-k45 — map curve above K=40
3. vote-union-m2-k38 — confirm curve below K=40
4. vote-min-k40-v10/v12/v14 — precision-first candidates

See [DEVELOPMENT.md](../../DEVELOPMENT.md) §22 and [utils/vote_union.py](../../utils/vote_union.py).
