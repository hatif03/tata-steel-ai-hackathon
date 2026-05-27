# GBM Recall — Full Missing Indicators

Equal-weight XGB+LGB+Cat on **all 49 missing indicators** (`utils/tabular_features_full_miss.py`) vs 10 sparse columns in base gbm-recall.

Guarded FE: probed with canary guard on coils 806/1187 before use in union pool.

## How to run

```powershell
python models/gbm-recall-fullmiss/train.py --threshold-strategy top_k_33
python models/gbm-recall-fullmiss/predict.py
python scripts/probe_single_feature.py  # includes full_miss probe
```
