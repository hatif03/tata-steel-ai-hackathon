---
name: tata-steel-submission
description: Validates and prepares HackerEarth submission CSV files and source-code zip/tar archives for the Tata Steel hackathon. Use before uploading predictions, packaging approach.txt + source files, checking submission format, or fixing checker errors for Round 1.
---

# Tata Steel Submission

Problem page: https://www.hackerearth.com/challenges/competitive/tata-steel-ai-hackathon/machine-learning/fd-a5a6dcb2/

HackerEarth requires **two uploads**:

1. **Prediction file** — CSV matching `dataset/sample_submission.csv`
2. **Source archive** — zip or tar containing a **text file** (approach, feature engineering, tools) plus **relevant source files**

Each method folder keeps both under `models/{method}/submission/`.

## Submission folder layout

```
models/{method}/submission/
  approach.txt                      # you write: approach, FE, tools, reproduce steps
  submission.csv                    # predictions (CoilID,Y) — from latest run
  source/                           # auto-copied by pack script
    {method}.ipynb
    train.py
    predict.py
    README.md
    requirements.txt
    utils/...
  {method}-hackerearth.zip          # upload this as source bundle
  pack_meta.json                    # timestamp + paths (optional audit)
```

## Prediction file requirements

| Rule | Detail |
|------|--------|
| Format | CSV with header `CoilID,Y` |
| Rows | Exactly one per test `CoilID` (339 rows) |
| IDs | Must match `dataset/test.csv` exactly |
| Values | `Y` ∈ {0, 1} integers — not probabilities |
| Duplicates | None |
| Extras | No extra columns or rows |

## Full workflow

```bash
# 1. Train and predict
python models/{method}/train.py
python models/{method}/predict.py

# 2. Edit submission/approach.txt (required before first pack)

# 3. Package + validate
python .cursor/skills/tata-steel-submission/scripts/pack_submission.py models/{method}
```

Tar instead of zip:

```bash
python .cursor/skills/tata-steel-submission/scripts/pack_submission.py models/{method} --format tar.gz
```

Use a specific run:

```bash
python .cursor/skills/tata-steel-submission/scripts/pack_submission.py models/{method} \
  --run-dir models/{method}/outputs/runs/20260525_174159
```

## Validate only (no packaging)

```bash
python .cursor/skills/tata-steel-submission/scripts/validate_submission.py \
  models/{method}/submission/submission.csv
```

Or from a run directory before packing:

```bash
python .cursor/skills/tata-steel-submission/scripts/validate_submission.py \
  models/{method}/outputs/latest/predictions/submission.csv
```

## approach.txt (required content)

Plain text (not PDF). Include:

1. **Approach** — model family, validation, threshold/imbalance strategy
2. **Feature engineering** — raw vs derived features, exclusions (e.g. CoilID), missing-value handling
3. **Tools** — Python version, libraries with versions if pinned, Colab vs local
4. **Reproduction** — commands to regenerate `submission.csv` from `dataset/`

Keep in sync with the method `README.md`; the archive ships `approach.txt` for reviewers.

## Git (root `.gitignore`)

| Committed | Gitignored (regenerate locally) |
|-----------|----------------------------------|
| `submission/approach.txt` | `submission/source/` |
| `submission/README.md` | `submission/submission.csv` |
| Source code, notebooks | `submission/*-hackerearth.zip` |
| | `submission/pack_meta.json` |
| | `models/*/outputs/` |

## Pre-upload checklist

```
Submission checklist:
- [ ] approach.txt updated in models/{method}/submission/
- [ ] pack_submission.py succeeds (validates CSV + builds zip)
- [ ] Upload submission/submission.csv (predictions)
- [ ] Upload submission/{method}-hackerearth.zip (source)
- [ ] Notebook reproduces submission end-to-end
- [ ] Random seed documented
- [ ] No external data used
- [ ] CoilID not used as model feature
- [ ] Leaderboard score logged in method README
```

## Generate submission (standard)

```python
import pandas as pd

test = pd.read_csv("dataset/test.csv")
submission = pd.DataFrame({"CoilID": test["CoilID"], "Y": y_pred.astype(int)})
submission.to_csv(run_dir / "predictions/submission.csv", index=False)
```

Then run `pack_submission.py` to copy into `submission/submission.csv` and build the archive.

## After upload

1. Record public leaderboard score in the method README **Results** section
2. If score drops vs CV, check threshold, imputation on test, and ID alignment
3. Re-run `pack_submission.py` after any change you submit again

## Common checker failures

| Error symptom | Fix |
|---------------|-----|
| Wrong row count | Ensure len(submission) == len(test) |
| Missing IDs | Merge against test CoilID list |
| Extra IDs | Inner-join to test only |
| Float predictions | `.astype(int)` after threshold |
| Probabilities uploaded | Apply threshold; submit 0/1 only |
| Missing approach.txt | Create before running pack_submission.py |
