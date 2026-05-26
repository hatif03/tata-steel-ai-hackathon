# GBM Recall Ensemble

Equal-weight **XGBoost + LightGBM + CatBoost** with recall-first threshold (forum-aligned).

Unlike [`gbm-ensemble`](../gbm-ensemble/README.md) (failed LB 1.13208 with t=0.72), this uses **equal 1/3 blend weights** and `select_recall_oriented_threshold`.

## Results

| Upload | LB score | Test pos | Strategy |
|--------|----------|----------|----------|
| forum_fixed / top_k_33 | **12.45283** | 33 | t=0.05 ≡ rank-K=33 on equal blend |
| Phase 6 rethreshold | 9.05660 | 24 | t≈0.121 |
| target_fpr (early) | 5.66038 | 15 | t≈0.24 |

| Metric (latest top_k_33 run) | Value |
|--------|-------|
| OOF PR-AUC | 0.360 |
| OOF accuracy @ K=33 | 0.950 |
| Canary coils | all Y=1 |

**Phase 8 note:** gbm-recall @ 33 positives is the **anchor set** for union augments. Adding secondary-model exclusives via [`build_union_submission.py`](../../scripts/build_union_submission.py) beat this score: **13.21** (K=35) and **14.34** (K=38). See [`models/phase8-rethreshold/README.md`](../phase8-rethreshold/README.md).

## How to run

```powershell
python models/gbm-recall/train.py --threshold-strategy top_k_33
python models/gbm-recall/predict.py
python scripts/rethreshold_submission.py --phase8 --methods gbm-recall --write-all
python models/gbm-recall/pack.py
```
