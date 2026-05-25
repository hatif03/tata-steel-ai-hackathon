# Tata Steel Hackathon — Reference

## Links

- Hackathon hub: https://www.hackerearth.com/community/challenges/competitive/tata-steel-ai-hackathon/
- ML problem: https://www.hackerearth.com/challenges/competitive/tata-steel-ai-hackathon/machine-learning/fd-a5a6dcb2/
- HackerEarth checker docs: https://help.hackerearth.com/hc/en-us/articles/900005034803-checker-files

## Timeline (2026 edition)

| Phase | Dates |
|-------|-------|
| Round 1 ML challenge | May 22 – May 31, 2026 |
| Round 2 Advanced AI | June 5 – June 15, 2026 |
| Finals | June 22 – June 26, 2026 |

Confirm dates on the official HackerEarth page before planning.

## Submission deliverables

1. **Prediction file** — CSV matching `sample_submission.csv` format (`CoilID,Y`)
2. **Source archive** — zip/tar with a **text file** (approach, feature engineering, tools) plus relevant source files

Each method folder stores both under `models/{method}/submission/`:

```
submission/
  approach.txt
  submission.csv
  {method}-hackerearth.zip
```

Package with:

```bash
python .cursor/skills/tata-steel-submission/scripts/pack_submission.py models/{method}
```

Multiple submissions typically allowed during Round 1; keep a log of leaderboard scores per submission.

## Feature columns

All features are numeric (`float64`). Naming only — domain semantics are not provided in the dataset:

| Group | Columns | Notes |
|-------|---------|-------|
| Process / continuous | X1–X33 | Wide dynamic range; likely sensor or process parameters |
| Count / flag-like | X34–X49 | Many zeros; integer-like values mixed with small decimals |

### Missing value counts (train)

| Column | Null count |
|--------|------------|
| X15 | 160 |
| X42 | 31 |
| X48 | 13 |
| X26, X16, X24, X10, X25, X27, X23 | 6 each |
| Others | 0–5 |

193 rows contain at least one missing value (14.3% of train).

## Target

```
Y=0: 1286 (95.1%)
Y=1:   66 ( 4.9%)
```

## Submission schema

```csv
CoilID,Y
711,0
1542,0
...
```

- Header required: `CoilID,Y`
- One row per test `CoilID`; no extras, no duplicates
- `Y` must be integer 0 or 1 (not probabilities)

## Accuracy baseline math

Majority class (all 0): 1286/1352 ≈ **95.12%** on train class distribution.

To win, models must recover true positives without flooding false positives. Useful diagnostic:

```python
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

# threshold sweep on OOF probabilities
for t in np.linspace(0.05, 0.95, 19):
    pred = (proba >= t).astype(int)
    print(t, accuracy_score(y_val, pred), f1_score(y_val, pred))
```

## Industrial context (hypothesis)

Coil-level tabular data with rare positive labels fits **quality/defect/anomaly** prediction on steel production lines. Feature engineering should preserve interpretability for potential Round 2 / interview discussions.

## Hiring context

Top performers may receive PPI opportunities and joining incentives (verify on official page). Clean, documented, reproducible code matters alongside leaderboard rank.
