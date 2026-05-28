# GBM Recall Safe FE

Equal-weight XGB + LightGBM + CatBoost with **ratio_X13_X7** (only ratio passing canary guard in Phase 10 probe). Top-K=33 output.

## How to run

```powershell
python models/gbm-recall-safe-fe/train.py --threshold-strategy top_k_33
python models/gbm-recall-safe-fe/predict.py
python models/gbm-recall-safe-fe/pack.py
```

Probe report: `models/phase8-rethreshold/outputs/shap_ratio_probe_report.json`
