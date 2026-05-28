# AutoGluon Recall

AutoGluon `TabularPredictor` with configurable preset, OOF via fold-wise models, top-K recall threshold.

## How to run (local)

```powershell
pip install autogluon
python models/autogluon-recall/train.py --k 33 --time-limit 600 --preset medium_quality
python models/autogluon-recall/predict.py
python models/autogluon-recall/pack.py
```

## Colab (best_quality — recommended)

```python
%pip install autogluon
import os
os.chdir("/content/tata-steel-ai-hackathon")  # clone repo first
!python models/autogluon-recall/train.py --preset best_quality --time-limit 3600 --k 33
!python models/autogluon-recall/predict.py
```

Copy `models/autogluon-recall/outputs/latest/predictions/test_predictions.csv` back to local, then rebuild vote union:

```powershell
python scripts/build_vote_union.py --k-values 40 41 42 --min-votes 2
```

## Notes

- Use `--preset best_quality` on Colab; local default is `medium_quality` (lower RAM).
- Joins union pool as secondary model after predict.
