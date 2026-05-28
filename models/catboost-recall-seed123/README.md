# CatBoost Recall — seed 123

CatBoost classifier variant with `random_seed=123` for vote-union pool diversity (Phase 11).

## Method

- Same pipeline as `catboost-recall`: median imputation via `utils/tabular_features`, stratified 5-fold CV, recall-first top-K threshold.
- **Seed:** 123 (distinct proba landscape vs default `catboost-recall` @ seed 42).
- **Hyperparameters:** iterations=800, depth=5, lr=0.03, subsample=0.8, `auto_class_weights=Balanced`.

## Why this approach

Vote union at K=40 benefits from diverse model probas. A second CatBoost seed adds another top-K vote without retraining the full factory.

## How to run

```powershell
python models/catboost-recall-seed123/train.py --data-dir dataset
python models/catboost-recall-seed123/predict.py --data-dir dataset
python models/catboost-recall-seed123/pack.py
```

Then rebuild vote union:

```powershell
python scripts/build_vote_union.py --k-values 40 41 42 --min-votes 2
```

## Results

See `outputs/latest/metrics.json` after training.
