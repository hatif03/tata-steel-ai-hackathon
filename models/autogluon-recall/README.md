# AutoGluon Recall

AutoGluon `TabularPredictor` with `presets=best_quality`, OOF via fold-wise medium_quality models, top-K recall threshold.

## How to run

```powershell
pip install autogluon
python models/autogluon-recall/train.py --k 33 --time-limit 600
python models/autogluon-recall/predict.py
python models/autogluon-recall/pack.py
```

## Notes

- Full fit uses best_quality; OOF uses faster medium_quality per fold for speed.
- Joins union pool as secondary model after predict.
