# Meta Recall Stack

LogisticRegression / weight-opt meta-learner on OOF probas from 5–9 recall base models. Selects strategy and K ∈ {33, 35, 38, 40} by OOF accuracy @ top-K.

## How to run

```powershell
python models/meta-recall-stack/train.py
python models/meta-recall-stack/predict.py
python models/meta-recall-stack/pack.py
```

## Technical details

- Sources (skip if missing): sklearn-recall, lightgbm-recall, gbm-recall, recall-blend, rf-smote-v2, mega-recall-blend, lightgbm-optuna, gbm-mega-blend, catboost-recall
- Strategies: equal_weight, weight_opt (Nelder-Mead on OOF acc @ K), stacking (LogisticRegression)
- Canary coils forced in top-K
