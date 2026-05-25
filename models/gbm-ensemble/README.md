# GBM Ensemble

Weighted blend of **XGBoost**, **LightGBM**, and **CatBoost** on shared tabular features with missingness indicators. Builds on the LightGBM single-model submission (HackerEarth score **1.88679**).

## Method

- **Models**: `XGBClassifier` + `LGBMClassifier` + `CatBoostClassifier`
- **Blend**: OOF probability weighted average; weights tuned on train OOF
- **Features**: `utils/tabular_features.py` — raw `X1`–`X49` (NaN kept) + `miss_*` + `row_missing_count`
- **CV**: Stratified 5-fold, `random_state=42`
- **Threshold**: OOF accuracy maximization on blended probabilities

### Optimized blend weights (latest run)

| Model | Weight |
|-------|--------|
| LightGBM | 0.45 |
| CatBoost | 0.35 |
| XGBoost | 0.20 |

### Hyperparameters (all bases)

~500 trees, depth 4, learning rate 0.05, subsample 0.8. XGB/LGBM: `scale_pos_weight` ≈ 19.48. CatBoost: `auto_class_weights='Balanced'`.

## Why this approach

Single GBMs make different errors on 1,352 imbalanced rows. LightGBM led the leaderboard; adding CatBoost (ordered boosting, balanced class weights) and XGBoost (different split logic) with **OOF-tuned blend weights** improves PR-AUC and accuracy without external data.

## Technical details

- Same leakage guards as prior methods: no `CoilID`, no test labels, fold-wise OOF only for weight/threshold tuning.
- Weight search: grid over simplex (step 0.05), joint with threshold sweep for accuracy.
- Artifacts: three serialized models + `meta.joblib` (weights, threshold, feature list).

## Results

| Metric | **gbm-ensemble** | lightgbm-cv | xgboost-baseline |
|--------|-------------------|-------------|------------------|
| OOF PR-AUC | **0.3631** | 0.3541 | 0.3269 |
| OOF accuracy | **96.08%** (t=0.72) | 95.86% (t=0.73) | 95.56% (t=0.81) |
| Positive recall (OOF) | **22.7%** (15/66) | 21.2% | 13.6% |
| OOF positives predicted | 17 | 18 | 12 |
| Test positives predicted | 3 | 5 | 3 |
| HackerEarth score | **1.13208** (regressed) | **1.88679** | 1.13208 |

Per-model OOF PR-AUC: XGB 0.330, LGBM 0.354, CatBoost 0.323.

## How to run

```powershell
.\.venv\Scripts\Activate.ps1
python models/gbm-ensemble/train.py
python models/gbm-ensemble/predict.py
python models/gbm-ensemble/pack.py
```

Upload `models/gbm-ensemble/submission/submission.csv` and `gbm-ensemble-hackerearth.zip` to HackerEarth.
