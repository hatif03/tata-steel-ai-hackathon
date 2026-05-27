# CatBoost Recall

Single CatBoostClassifier with native NaN handling and top-K=33 recall-first threshold.

## How to run

```powershell
python models/catboost-recall/train.py --k 33
python models/catboost-recall/predict.py
python models/catboost-recall/pack.py
```

Used as a secondary source in union augment and meta-recall-stack.
