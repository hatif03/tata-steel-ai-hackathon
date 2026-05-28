# kNN Positive Profile

Scores each test coil by **maximum cosine similarity** to the train Y=1 manifold (66 positives). Top-K=33 with canary forcing.

## Method

- Median impute + StandardScaler inside CV
- Score = 1 − min cosine distance to any train positive
- Complements tree models for union secondary pool

## How to run

```powershell
python models/knn-positive-profile/train.py
python models/knn-positive-profile/predict.py
python models/knn-positive-profile/pack.py
```

Optional Intel GPU via scikit-learn-intelex for imputer/scaler (`--cpu-only` to disable).

## Results

See `outputs/latest/metrics.json` after training.
