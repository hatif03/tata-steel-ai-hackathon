# Mega Recall Blend

Combines **sklearn-recall + lightgbm-recall + gbm-recall** probas via weight optimization, stacking, or vote rules. Selection by OOF accuracy at rank-based **top-K** (default K=26) with canary coils forced positive.

## Strategies evaluated

- `weight_opt` — scipy Nelder-Mead weights on OOF probas
- `stacking` — LogisticRegression meta-learner
- `equal_weight`, `majority_vote`, `union`, `intersection`

## How to run

```powershell
python models/mega-recall-blend/train.py --k 26
python models/mega-recall-blend/predict.py
python scripts/rethreshold_submission.py --methods mega-recall-blend --write-all
python models/mega-recall-blend/pack.py
```

## Results

| Metric | Value |
|--------|-------|
| Selected strategy | intersection (test); top_k_26 OOF |
| K | 26 |
| OOF accuracy | 0.956 |
| Test positives | 26 |
