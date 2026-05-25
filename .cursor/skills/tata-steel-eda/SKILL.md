---
name: tata-steel-eda
description: Runs exploratory data analysis on the Tata Steel hackathon dataset (train.csv, X1–X49, imbalanced Y). Use when analyzing features, missing values, correlations, class imbalance, leakage checks, or deciding preprocessing before modeling.
---

# Tata Steel EDA

## Goals

1. Confirm schema and row counts match expectations
2. Quantify imbalance and establish baselines
3. Profile missingness (especially X15)
4. Find redundant / low-variance features
5. Surface separability hints without overfitting
6. Document findings in method README or a shared notebook under `utils/` or `notebooks/`

## Standard analysis script

Run from repo root:

```bash
python .cursor/skills/tata-steel-eda/scripts/profile_dataset.py
```

Read stdout; save key numbers to the active method's README **Results** or **Technical details** section.

## Manual EDA checklist

Copy and track:

```
EDA Progress:
- [ ] Shape & dtypes
- [ ] Target distribution + baseline accuracy
- [ ] Missing values (per column + per row)
- [ ] Train/test ID disjointness
- [ ] Feature scale & outliers (describe, boxplot logic)
- [ ] Correlation / redundancy
- [ ] Univariate signal (PR-AUC per feature)
- [ ] Leakage sanity checks
```

### 1. Load and validate

```python
import pandas as pd

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")
assert len(set(train.CoilID) & set(test.CoilID)) == 0
features = [c for c in train.columns if c.startswith("X")]
```

### 2. Target and baselines

```python
print(train["Y"].value_counts(normalize=True))
# Majority baseline accuracy ≈ 0.951
```

Always report **PR-AUC** alongside accuracy — it reflects rare-class detection.

### 3. Missingness

- Count nulls per column; flag X15 (~12% of rows)
- Check whether `Y=1` rows have higher missing rates (informative missingness)
- Plan: impute inside CV; consider `is_missing_X15` indicator

### 4. Feature groups

| Group | Columns | EDA focus |
|-------|---------|-----------|
| X1–X33 | Continuous | distributions, pairs, PCA |
| X34–X49 | Count/flag | zero inflation, value counts |

### 5. Redundancy

- Pearson correlation matrix on imputed train data
- Drop pairs with |r| > 0.95 from some models (not necessarily all)
- Check near-zero variance columns

### 6. Univariate predictive power

```python
from sklearn.metrics import average_precision_score
from sklearn.impute import SimpleImputer

X = SimpleImputer(strategy="median").fit_transform(train[features])
y = train["Y"].values
for i, col in enumerate(features):
    # rank by single-feature score as sanity check
    pass  # sort features by PR-AUC
```

### 7. Leakage guards

- Never merge train statistics into test before split
- Do not use `CoilID` as a feature
- If adding target-encoded features later, use nested CV only

## Outputs to produce

After EDA, write a short summary:

```markdown
## EDA summary
- Train/test: 1352 / 339; IDs disjoint
- Pos rate: 4.9%; majority baseline acc 95.1%
- Missing: 193 rows; X15 worst (160)
- Top univariate signals: X?, X?, ...
- Dropped / flagged: ...
- Preprocessing plan: median impute + missing flags; stratified 5-fold CV
```

## Next step

Scaffold a model experiment: see [tata-steel-model-experiment](../tata-steel-model-experiment/SKILL.md).
