# RF + SMOTE v2

RandomForest + BorderlineSMOTE inside CV **without** class_weight double penalty. Threshold via rank-based **top-K=26** (not literal forum t=0.31).

## How to run

```powershell
python models/rf-smote-v2/train.py --k 26
python models/rf-smote-v2/predict.py
python models/rf-smote-v2/pack.py
```

## Results

| Metric | Value |
|--------|-------|
| OOF PR-AUC | ~0.17 |
| Test positives @ top_k_26 | 26 |
| Canary-safe @ rank-K | yes |
