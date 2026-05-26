# RF + SMOTE (Forum Pattern)

**RandomForest** with **SMOTE applied only on training folds** (imblearn Pipeline + StratifiedKFold). Forum recipe: threshold **0.31**, ~26/339 test positives.

## Why this approach

Forum winners reported RF+SMOTE with t=0.31 and ~26 positives for leaderboard ~100. SMOTE is strictly inside CV training folds to avoid leakage.

## Technical details

- Median impute → SMOTE (k=5) → RF (500 trees, max_depth=8, class_weight {0:1, 1:30})
- Default threshold: fixed t=0.31; sweep 0.25–0.35 via Phase 6 scripts
- Features: `utils/tabular_features.py`

## How to run

```powershell
pip install imbalanced-learn
python models/rf-smote/train.py
python models/rf-smote/predict.py
python scripts/check_submission_vs_baseline.py models/rf-smote/outputs/latest/predictions/submission.csv
python models/rf-smote/pack.py
```

## Results

| Metric | Value |
|--------|-------|
| OOF PR-AUC | 0.171 |
| Forum t=0.31 test positives | 131 (over-predicts) |
| 24–28 band | No canary-safe threshold found |

Forum fixed t=0.31 does not transfer to this dataset; use Phase 6 rethreshold sweep if revisiting.
