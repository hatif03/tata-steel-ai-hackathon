---
name: tata-steel-hackathon
description: Guides work on the Tata Steel AI Hackathon tabular ML challenge (CoilID, X1–X49, binary Y). Use when building models, running EDA, creating submissions, or discussing hackathon strategy, HackerEarth leaderboard, PPI hiring, or winning approaches for this competition.
---

# Tata Steel AI Hackathon

## Problem (Round 1 — ML Challenge)

| Item | Detail |
|------|--------|
| Platform | [HackerEarth](https://www.hackerearth.com/community/challenges/competitive/tata-steel-ai-hackathon/) |
| Problem page | [ML challenge](https://www.hackerearth.com/challenges/competitive/tata-steel-ai-hackathon/machine-learning/fd-a5a6dcb2/) |
| Task | Binary classification: predict `Y` from 49 numeric features per steel coil |
| ID column | `CoilID` — identifier only; **never use as a feature** |
| Data location | `dataset/train.csv`, `dataset/test.csv`, `dataset/sample_submission.csv` |
| Rules | **No external data**; individual participation; submit predictions + source code (zip/tar) |

## Dataset snapshot

| Split | Rows | Columns |
|-------|------|---------|
| Train | 1,352 | CoilID + X1–X49 + Y |
| Test | 339 | CoilID + X1–X49 |

- **Class imbalance**: Y=1 is ~4.9% (66 positives / 1,286 negatives). Treat as rare-event detection.
- **Missing values**: 193 train rows have ≥1 missing feature; **X15** is sparsest (~160 nulls). Missingness may be informative — impute inside CV folds only.
- **No ID leakage**: train and test `CoilID` sets are disjoint.

Full column stats and rules: [reference.md](reference.md)

## Evaluation and winning strategy

Public sources indicate Round 1 is scored primarily on **classification accuracy** via HackerEarth's checker. With ~95% negatives, a naive all-zero predictor scores ~95% — **you must beat that baseline meaningfully**.

### Validation vs leaderboard

| Monitor locally | Optimize for submission |
|-----------------|---------------------------|
| PR-AUC, F1, balanced accuracy | Accuracy at chosen threshold |
| Stratified K-fold (k=5) | Match HackerEarth binary `{0,1}` predictions |

After CV, tune the **decision threshold** on out-of-fold probabilities to maximize expected accuracy (not default 0.5).

### High-impact modeling choices

1. **Gradient boosting first** — LightGBM, XGBoost, CatBoost handle mixed scales, missing values, and imbalance (`scale_pos_weight`, `class_weight`, focal loss variants).
2. **Class imbalance** — stratified splits; prefer PR-AUC for model selection; threshold-tune for accuracy.
3. **Ensembling** — average or rank-average probabilities from 2–3 diverse GBMs; often +0.5–2% accuracy on small tabular sets.
4. **Feature work** — ratios/interactions among X1–X33 (continuous process signals) and X34–X49 (counts/flags); missing indicators for X15; avoid target encoding without nested CV.
5. **Reproducibility** — fix seeds (`random_state=42`); log library versions in notebooks.

### Anti-patterns

- Using external datasets or pretrained models trained on outside steel/defect data
- Tuning on test predictions or peeking at test labels
- Optimizing PR-AUC alone without threshold calibration for accuracy
- Putting `CoilID` in the feature matrix
- Single holdout split on 1,352 rows (high variance)

## Repo conventions

Experiments live under `models/{method-name}/` with `README.md` + `{method-name}.ipynb`. Shared helpers go in `utils/`. The always-on rule `.cursor/rules/tata-steel-ml-methods.mdc` defines folder layout — follow it for every new approach.

**Local runs** save metrics, plots, artifacts, and submissions under `models/{method}/outputs/runs/<timestamp>/` with a copied summary in `outputs/latest/`.

## Workflow map

```
EDA & baseline ──► new method folder ──► CV + threshold tune ──► validate submission ──► upload
       │                    │                      │                      │
  tata-steel-eda    tata-steel-model-experiment   (in method nb)   tata-steel-submission
```

## Related skills

| Skill | When |
|-------|------|
| [tata-steel-eda](../tata-steel-eda/SKILL.md) | First look at data, missingness, feature relationships |
| [tata-steel-model-experiment](../tata-steel-model-experiment/SKILL.md) | Scaffold or extend a model under `models/` |
| [tata-steel-submission](../tata-steel-submission/SKILL.md) | Generate and validate `submission.csv` before upload |

## Quick baseline checklist

- [ ] Load train/test; drop `CoilID` from features
- [ ] Stratified 5-fold CV; report PR-AUC + accuracy at tuned threshold
- [ ] Compare to majority-class baseline (~95.1% accuracy)
- [ ] Fit final model on full train; predict test
- [ ] Run submission validator; upload to HackerEarth
- [ ] Confirm `outputs/latest/metrics.json` and plots saved in repo
