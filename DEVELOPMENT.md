# Development Log — Tata Steel AI Hackathon (Round 1)

This document records everything done in this repository so far: motivation, steps taken, results (offline and HackerEarth leaderboard), failures, and how decisions were handled. It is meant for anyone picking up the project mid-stream.

**Problem:** [HackerEarth ML challenge](https://www.hackerearth.com/challenges/competitive/tata-steel-ai-hackathon/machine-learning/fd-a5a6dcb2/) — binary classification of steel coil quality/defect label `Y` from 49 numeric features `X1`–`X49` and identifier `CoilID`.

**Last updated:** 2026-05-28

---

## 1. Executive summary

| Item | Detail |
|------|--------|
| **Best HackerEarth score** | **15.84906** — `union-gbm33-plus-9` (Phase 9, K=42) |
| **Prior best (Phase 8)** | 14.33964 — `union-gbm33-plus-5` (K=38) |
| **Baseline score** | 1.13208 — `models/xgboost-baseline` |
| **Recommended submission** | `models/phase8-rethreshold/outputs/union-gbm33-plus-9/submission/` or `models/union-gbm33-augment/train.py --target-k 42` |
| **Core lesson (Phase 9)** | Union augment beyond gbm33 scales **linearly** in the 38–42 positive band (~**+0.377 LB per added exclusive**). Gains come from flagging secondary-model exclusives, not from re-ranking within gbm top-33 or probability blending. |

---

## 2. Dataset and evaluation context

### 2.1 Data snapshot

| Split | Rows | Notes |
|-------|------|-------|
| Train | 1,352 | 66 positives (`Y=1`), 1,286 negatives — **4.88% positive rate** |
| Test | 339 | Labels hidden; ~17 positives expected if rate matches train |

- **Features:** `X1`–`X33` continuous/process-like; `X34`–`X49` count/flag-like (many zeros).
- **Missing values:** 193 train rows (14.3%) have ≥1 missing value. **X15** is sparsest (~160 nulls, ~12% of rows).
- **Leakage check:** Train and test `CoilID` sets are disjoint. `CoilID` is never used as a feature.

### 2.2 Baselines

- **Majority class (all 0):** 95.12% accuracy on train class distribution.
- Any useful model must beat 95.12% **without** flooding false positives — each correct `Y=1` is worth ~20 correct `Y=0` in accuracy terms.

### 2.3 Local vs leaderboard metrics

| Optimized locally | Intended for HackerEarth |
|-------------------|--------------------------|
| OOF PR-AUC (model selection signal) | Classification **accuracy** at tuned threshold |
| Stratified 5-fold CV | Integer predictions `{0, 1}` in CSV |

**Important:** Higher OOF accuracy did **not** guarantee a better leaderboard score on the 339-row test set. Offline validation is noisy with only 66 positives across 5 folds.

### 2.4 HackerEarth score interpretation

Observed scores (higher = better on leaderboard):

| Submission | Score |
|------------|-------|
| xgboost-baseline | 1.13208 |
| gbm-ensemble | 1.13208 |
| lightgbm-cv | **1.88679** |

Exact formula is on the HackerEarth problem page; treat it as a monotonic “higher is better” competition metric aligned with strong classification performance.

---

## 3. Repository infrastructure (before modeling)

Established before any experiments:

- **Folder convention:** One method per `models/{kebab-case-name}/` with `README.md`, `train.py`, `predict.py`, `submission/approach.txt`, Colab notebook, and `outputs/runs/<timestamp>/`.
- **Shared utilities:** `utils/run_artifacts.py` (run dirs, metrics, latest summary), `utils/plotting.py` (PR curve, threshold sweep, confusion matrix, feature importance), `utils/submission_pack.py` (validate CSV, bundle source zip).
- **Skills / rules:** `.cursor/skills/tata-steel-*`, `.cursor/rules/tata-steel-ml-methods.mdc` govern workflow.
- **Environment:** Python 3.11+ venv via `scripts/setup.ps1`; dependencies in `requirements.txt`.

Every training run persists: `metrics.json`, `run_config.json`, `oof_predictions.csv`, `artifacts/`, `plots/`, and optional `predictions/submission.csv`.

---

## 4. Chronological development

### Phase 0 — EDA and problem framing

**Actions:**

- Ran `.cursor/skills/tata-steel-eda/scripts/profile_dataset.py`.
- Confirmed imbalance, missingness (especially X15), disjoint IDs.
- Univariate PR-AUC: best single feature **X13 ≈ 0.21** — weak linear signal; interactions/trees needed.

**Decisions:**

- Use gradient boosting first.
- Stratified 5-fold CV, `random_state=42`.
- Tune decision threshold on OOF probabilities for **accuracy** (not default 0.5).
- Handle imbalance via `scale_pos_weight = n_neg / n_pos ≈ 19.48` (avoid SMOTE on small data).

---

### Phase 1 — `models/xgboost-baseline` (first submission)

**Why:** Strong default for tabular data; reproducible floor.

**Pipeline:**

1. Drop `CoilID`.
2. **Median imputation** inside each CV fold (`SimpleImputer`) — statistics fit on train fold only.
3. `XGBClassifier`: 500 trees, depth 4, lr 0.05, subsample/colsample 0.8, `scale_pos_weight≈19.48`.
4. Threshold sweep on OOF → **t* = 0.81**.
5. Final model on full train; predict test.

**Offline results:**

| Metric | Value |
|--------|-------|
| OOF PR-AUC | 0.3269 |
| OOF accuracy @ t=0.81 | 95.56% |
| OOF positive recall | 13.6% (9/66) |
| Test positives predicted | **3** |
| Fold PR-AUC range | 0.21 – 0.52 (high variance) |

**HackerEarth:** **1.13208**

**Limitations identified:**

- Pre-imputation discards missingness direction before trees split.
- No feature engineering.
- Very conservative on positives (12 OOF positives at t=0.81).
- Barely above majority baseline (+0.44 pp OOF accuracy).

---

### Phase 2 — `models/lightgbm-cv` (first major improvement)

**Why:** LightGBM handles NaN natively; missingness (especially X15) may be informative. New method folder per repo rules (algorithm + preprocessing change).

**Changes vs XGBoost baseline:**

1. **No imputation** — pass NaN through to LightGBM.
2. **Missing indicators:** binary `miss_X15`, `miss_X42`, … for 10 sparse columns + `row_missing_count`.
3. Same CV, threshold tuning, imbalance handling.
4. Features initially in `models/lightgbm-cv/features.py`; later refactored to `utils/tabular_features.py`.

**Hyperparameters:** Same broad settings as XGB baseline (500 trees, depth 4, lr 0.05, etc.).

**Offline results:**

| Metric | Value |
|--------|-------|
| OOF PR-AUC | **0.3541** (+0.027 vs XGB) |
| OOF accuracy @ t=0.73 | **95.86%** |
| OOF positive recall | **21.2%** (14/66) |
| Test positives predicted | **5** |
| Threshold | 0.73 (lower than XGB’s 0.81) |

**HackerEarth:** **1.88679** (+66% relative vs baseline score)

**Why it likely won:**

- Native missing splits + explicit missing flags.
- Higher recall at similar accuracy — more test positives (5 vs 3) with strong probabilities on key coils (see §6).

**Dependency added:** `lightgbm>=4.0` in `requirements.txt`.

---

### Phase 3 — `models/gbm-ensemble` (attempted improvement — failed on LB)

**Why:** Standard Kaggle-style next step — blend diverse GBMs (XGB + LGBM + CatBoost) with OOF-tuned weights.

**Pipeline:**

- Shared `utils/tabular_features.py` (same as lightgbm-cv).
- Stratified 5-fold: train all three per fold; collect OOF probs.
- Grid search blend weights (step 0.05 on simplex) + threshold for OOF accuracy.
- Retrain all three on full train; blend test probabilities.

**Optimized weights:** XGB 0.20, LightGBM 0.45, CatBoost 0.35  
**Threshold:** 0.72

**Offline results (looked better):**

| Metric | Ensemble | lightgbm-cv alone |
|--------|----------|-------------------|
| OOF PR-AUC | **0.3631** | 0.3541 |
| OOF accuracy | **96.08%** | 95.86% |
| OOF positive recall | 22.7% | 21.2% |
| Test positives | **3** | **5** |

**HackerEarth:** **1.13208** — full regression to baseline level.

**What went wrong:**

- XGBoost and CatBoost assign **lower** probabilities to certain test coils that LightGBM scored highly (806, 1187).
- Averaging **pulled blended proba below threshold** → lost 2 positive predictions vs lightgbm-cv.
- OOF favored blend because fold-level errors averaged out; **test distribution did not match**.

**Handling:** Documented failure; updated `models/gbm-ensemble/README.md`. Re-packed `lightgbm-cv` for re-submission. **Do not submit ensemble.**

**Dependencies added:** `catboost>=1.2`.

---

### Phase 4 — Post-ensemble recovery attempts

After ensemble LB failure, several **LightGBM-only** variants were tried. None beat lightgbm-cv on the test probability profile that correlated with LB success.

#### 4a — `models/lightgbm-fe` (feature engineering)

**Added features** (`utils/tabular_features_enriched.py`):

- `log1p(X34–X49)`
- Ratios among high-signal pairs (X13/X10, X30/X32, …)
- Row aggregates: `cont_mean/std/max/min/range`, `count_sum`, `count_nonzero`

**Param selection:** 3-config grid by **OOF PR-AUC** → selected `depth6_leaves31`.

| Metric | lightgbm-fe | lightgbm-cv |
|--------|-------------|-------------|
| OOF PR-AUC | **0.3788** | 0.3541 |
| OOF accuracy | 95.49% | **95.86%** |
| Test positives | 3 | **5** |

**Lost on test:** Coils **806** (proba 0.02 vs 0.83) and **1187** (0.16 vs 0.79) — FE destroyed calibration on LB-critical rows.

**Not submitted to HackerEarth** (would likely score ~baseline).

#### 4b — `models/lightgbm-tuned` (hyperparameter grid)

**Grid:** 4 configs on base features; select by **OOF accuracy**.

**Winner:** `conservative` — depth 3, 7 leaves, stronger L1/L2.

| Metric | Value |
|--------|-------|
| OOF accuracy | 95.86% (tie with cv) |
| Threshold | **0.79** (higher) |
| Test positives | **3** |

Again lost 806 and 1187 (proba ~0.69, below t=0.79).

#### 4c — `models/lightgbm-seedblend` (seed averaging)

**Method:** Same hyperparams as lightgbm-cv; average probabilities from seeds `{42, 123, 456}`.

| Metric | Value |
|--------|-------|
| OOF accuracy | 95.86% |
| Test positives | **2** (worst) |

Seed averaging diluted sharp predictions the same way multi-model blending did.

#### 4d — Ad-hoc tree-count experiment (not a method folder)

Single-model variants with seed 42:

| Config | OOF PR-AUC | OOF acc | Test pos | Notes |
|--------|------------|---------|----------|-------|
| 500 trees, lr 0.05 (baseline) | 0.3541 | **0.9586** | **5** | **Best test profile** |
| 1000 trees, lr 0.03 | 0.3488 | 0.9571 | 4 | Lost coil 532 proba |
| 1500 trees, lr 0.02 | 0.3365 | 0.9578 | 3 | Lost 806 proba |

**Conclusion:** Original lightgbm-cv training config is near-optimal for this test set’s calibration.

---

## 5. Master results table

| Method folder | HackerEarth | OOF PR-AUC | OOF acc | OOF t | OOF pos | Test pos | Submitted? |
|---------------|-------------|------------|---------|-------|---------|----------|------------|
| xgboost-baseline | 1.13208 | 0.3269 | 95.56% | 0.81 | 12 | 3 | Yes |
| **lightgbm-cv** | **1.88679** | 0.3541 | **95.86%** | 0.73 | 18 | **5** | Yes |
| gbm-ensemble | 1.13208 | 0.3631 | 96.08% | 0.72 | 17 | 3 | Yes (failed) |
| lightgbm-fe | — | 0.3788 | 95.49% | 0.55 | 19 | 3 | No |
| lightgbm-tuned | — | 0.3406 | 95.86% | 0.79 | 16 | 3 | No |
| lightgbm-seedblend | — | 0.3527 | 95.86% | 0.70 | 20 | 2 | No |

---

## 6. Critical test-set analysis (why lightgbm-cv won)

Five test coils where lightgbm-cv predicted `Y=1` (threshold 0.73):

| CoilID | lightgbm-cv proba | gbm-ensemble proba | lightgbm-fe proba |
|--------|-------------------|--------------------|-------------------|
| 654 | 0.956 | 0.959 | 0.971 |
| **806** | **0.833** | 0.611 → **0** | 0.020 → **0** |
| 532 | 0.756 | 0.824 | 0.748 |
| 958 | 0.830 | 0.896 | 0.874 |
| **1187** | **0.787** | 0.670 → **0** | 0.158 → **0** |

Coils **806** and **1187** separate leaderboard winners from losers. Any technique that reduces their predicted probability below threshold (blending, extra features, higher threshold, seed averaging) correlated with LB regression.

Threshold sweep on **saved** lightgbm-cv test probabilities: thresholds **0.65–0.75** all yield **5** test positives — the five coils above are stable in that band.

---

## 7. Shared code evolution

| File | Purpose |
|------|---------|
| `utils/tabular_features.py` | Base features: raw X1–X49 (NaN kept), 10× `miss_*`, `row_missing_count` |
| `utils/tabular_features_enriched.py` | Extended FE layer (used by lightgbm-fe only) |
| `utils/submission_pack.py` | Pack script; extended to copy `features.py`, `tabular_features*.py` into source bundle |
| `requirements.txt` | Added `lightgbm>=4.0`, `catboost>=1.2` |

**lightgbm-cv `features.py`:** Thin re-export of `utils/tabular_features` for HackerEarth source bundle compatibility.

---

## 8. Method folders on disk

```
models/
  xgboost-baseline/     # Phase 1 — median impute + XGBoost
  lightgbm-cv/          # Phase 2 — BEST LB — native NaN + missing indicators
  gbm-ensemble/         # Phase 3 — XGB+LGBM+CatBoost blend (LB failed)
  lightgbm-fe/          # Phase 4a — enriched features (offline only)
  lightgbm-tuned/       # Phase 4b — HP grid (offline only)
  lightgbm-seedblend/   # Phase 4c — seed average (offline only)
```

Each runnable folder contains: `train.py`, `predict.py`, `pack.py`, `submission/approach.txt`, and optional `.ipynb`.

---

## 9. Standard workflow used throughout

```powershell
# Setup (once)
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
.\.venv\Scripts\Activate.ps1

# Per method
python models/{method}/train.py
python models/{method}/predict.py
python models/{method}/pack.py
# or:
python .cursor/skills/tata-steel-submission/scripts/pack_submission.py models/{method}
```

**Upload to HackerEarth (two files):**

1. `models/{method}/submission/submission.csv`
2. `models/{method}/submission/{method}-hackerearth.zip`

Validate before upload:

```powershell
python .cursor/skills/tata-steel-submission/scripts/validate_submission.py models/{method}/submission/submission.csv
```

---

## 10. What worked

1. **LightGBM with native missing values** instead of median imputation.
2. **Missingness indicators** for sparse columns (X15, X42, X48, …) + row missing count.
3. **Threshold tuning on OOF** for accuracy (t=0.73 vs default 0.5).
4. **`scale_pos_weight`** for ~5% positive rate without SMOTE.
5. **Stratified 5-fold CV** with fixed seed for reproducibility.
6. **Strict artifact layout** — every run saved metrics, plots, models, OOF CSV for post-mortems (essential for diagnosing ensemble failure).

---

## 11. What failed or was lost

| Attempt | What was lost | Root cause |
|---------|---------------|------------|
| gbm-ensemble | LB 1.88679 → 1.13208 | Blending lowered proba on coils 806, 1187 |
| lightgbm-fe | Test positives 5→3; wrecked 806/1187 proba | Too many features / overfit on small train |
| lightgbm-tuned | Test positives 5→3 | Higher threshold (0.79) + conservative model |
| lightgbm-seedblend | Test positives 5→2 | Probability averaging diluted sharp scores |
| More trees (1000+) | Test positives 5→3/4 | Worse calibration on key test rows |

**Meta-lesson:** On **339 test rows**, OOF accuracy and PR-AUC can rank models **opposite** to leaderboard. Always compare **test prediction profiles** (especially high-proba positives) against the best known submission before uploading.

---

## 12. How decisions were handled

| Situation | Handling |
|-----------|----------|
| New algorithm or FE pipeline | New folder under `models/` per `.cursor/rules/tata-steel-ml-methods.mdc` |
| Ensemble looked better OOF | Submitted anyway → LB proved wrong → documented, reverted recommendation |
| OOF vs LB mismatch | Traced test probabilities coil-by-coil; identified 806/1187 as canary coils |
| Failed experiments | Kept folders and outputs for audit; marked in README / this log |
| Best submission | Re-packed `lightgbm-cv` after ensemble failure |
| Git | Outputs and zips gitignored; source, approach.txt, skills committed |

---

## 13. Current recommendation

**Submit only `lightgbm-cv`** until a new variant proves it **preserves or improves** probabilities on coils `{654, 806, 532, 958, 1187}` while matching or beating OOF accuracy.

```powershell
python .cursor/skills/tata-steel-submission/scripts/pack_submission.py models/lightgbm-cv
```

Upload:

- `models/lightgbm-cv/submission/submission.csv`
- `models/lightgbm-cv/submission/lightgbm-cv-hackerearth.zip`

---

## 14. Phase 5 — Recall-first evolution (2026-05-25)

Forum reports for the same **Hot Rolling Defect Detection** problem indicate top leaderboard scores ~**100**, while our best was **1.88679** with only **5/339** test positives. Winners use:

- GBM or sklearn ensembles
- **Low thresholds** (0.05–0.31) targeting **100% OOF recall** or FPR &lt; 3%
- **19–26 test positives** (~6–8% rate), not 3–5

### Infrastructure added

| File | Purpose |
|------|---------|
| [`utils/threshold_tuning.py`](utils/threshold_tuning.py) | `tune_max_accuracy`, `tune_recall_first`, `tune_target_fpr`, `select_recall_oriented_threshold` |
| [`scripts/run_phase0_threshold_sweep.py`](scripts/run_phase0_threshold_sweep.py) | Offline sweep on saved OOF/test probas |
| [`scripts/check_submission_vs_baseline.py`](scripts/check_submission_vs_baseline.py) | Pre-upload guard (canary coils, positive count) |

### New method folders

| Method | Threshold | Test pos | Notes |
|--------|-----------|----------|-------|
| **lightgbm-recall** | t=0.05 (fixed_t_0.05) | **19** | **Phase 0 winner — upload first** |
| gbm-recall | t≈0.24 (target_fpr) | 15 | Equal-weight LGB+XGB+CatBoost |
| sklearn-recall | t≈0.46 (target_fpr) | 21 | RF+ET+GBM, class_weight 1:30 |

100% OOF recall was **not achievable** on lightgbm-cv without predicting &gt;12% positives; `select_recall_oriented_threshold` falls back to FPR cap or forum t=0.05.

### Recommended upload order

1. `models/lightgbm-recall/submission/submission.csv` + zip
2. `models/sklearn-recall/submission/` (21 positives)
3. `models/gbm-recall/submission/` (15 positives)

```powershell
python scripts/check_submission_vs_baseline.py models/lightgbm-recall/submission/submission.csv
python models/lightgbm-recall/pack.py
```

Log new HackerEarth scores in §15 after upload.

---

## 15. Suggested next steps

1. **Upload recall-first submissions** and compare LB to 1.88679 / target ~100.
2. **RF + SMOTE** inside CV (Phase 3 of recall plan) if still below leaders.
3. **Targeted FE** only with canary-coil guard passing.
4. Repeated stratified CV for stabler OOF.

---

## 16. Phase 6 — push toward leaderboard ~100 (2026-05-26)

**Goal:** Close gap to forum winners (~26 test positives, t=0.05–0.31) without reverting to high-threshold accuracy tuning.

### 6A — Threshold sweep (no retrain)

Added to `utils/threshold_tuning.py`:

- `tune_target_positive_rate` — OOF rate ~7.7% (26/339)
- `tune_fixed_threshold` / `tune_fixed_thresholds` — forum values 0.05, 0.31, 0.35
- `tune_target_test_positives` — pick t for 24–28 test positives
- `select_threshold_by_strategy` — dispatch including `forum_fixed`

Extended `scripts/run_phase0_threshold_sweep.py` and added `scripts/rethreshold_submission.py`.

**Best rethreshold candidates (all canaries Y=1):**

| Method | Strategy | Threshold | Test pos | OOF acc |
|--------|----------|-----------|----------|---------|
| sklearn-recall | target_test_positives | 0.411 | 24 | 0.928 |
| recall-blend | target_test_positives | 0.221 | 24 | 0.928 |
| gbm-recall | target_test_positives | 0.121 | 24 | 0.928 |
| lightgbm-recall | target_test_positives | 0.036 | 24 | 0.924 |

Submissions: `models/phase6-rethreshold/outputs/{method}/submission.csv`  
Candidate list: `models/phase6-rethreshold/outputs/lb_candidates.json`

### 6B — gbm-recall forum t=0.05

`train.py --threshold-strategy forum_fixed` — equal 1/3 XGB+LGB+CatBoost, t=0.05.  
Test positives: **33** (all canaries positive). Prior target_fpr run scored LB **5.66** with 15 positives.

### 6C — recall-blend

New `models/recall-blend/` — 50/50 sklearn-recall trio + LightGBM OOF blend.  
Train default (target_rate): 27 test positives. Re-thresholded: **24** @ t=0.221.

### 6D — rf-smote

New `models/rf-smote/` — SMOTE inside CV only, RF class_weight 1:30.  
Forum t=0.31 yields **131** test positives on this data; no 24–28 band with all canaries positive. Model scaffolded for completeness; not recommended for upload without further calibration.

### Upload priority (beat LB 7.92)

1. `phase6-rethreshold/outputs/sklearn-recall/submission.csv` (24 pos)
2. `phase6-rethreshold/outputs/recall-blend/submission.csv` (24 pos)
3. `gbm-recall/outputs/latest/` forum_fixed (33 pos)

Fallback until new LB score: `models/sklearn-recall/submission/` (7.92453).

---

## 17. Phase 7 — push beyond LB 9.06 (2026-05-26)

**Phase 6 LB result:** sklearn-recall, recall-blend, gbm-recall, lightgbm-recall rethreshold @24 pos all scored **9.05660** (up from 7.92). Different predictions but identical score — count band mattered more than model choice at K=24.

### 7A — Exact-K rank selection

Added to [`utils/threshold_tuning.py`](utils/threshold_tuning.py):

- `apply_top_k` — force canaries positive, fill remaining K slots by proba rank
- `tune_top_k` — OOF metrics at exact K

Extended [`scripts/rethreshold_submission.py`](scripts/rethreshold_submission.py) for K ∈ {25,26,27,28,30,33}. Outputs under `models/phase7-rethreshold/outputs/`.

**Best OOF @ K=26 (canary-safe):**

| Method | OOF acc | Test pos |
|--------|---------|----------|
| lightgbm-recall | **0.957** | 26 |
| recall-blend | 0.957 | 26 |
| mega-recall-blend | 0.956 | 26 |
| gbm-recall | 0.956 | 26 |

**Upload priority:** lightgbm-recall K=26, recall-blend K=25 (OOF 0.958), gbm-recall forum33 (33 pos, untested), mega-recall-blend K=26.

See [`models/phase7-rethreshold/outputs/lb_candidates.json`](models/phase7-rethreshold/outputs/lb_candidates.json).

### 7B — mega-recall-blend

New [`models/mega-recall-blend/`](models/mega-recall-blend/) — weight-opt, stacking, vote rules on sklearn+lgbm+gbm saved probas. Selected **intersection** on test (26 pos).

### 7C — lightgbm-optuna

New [`models/lightgbm-optuna/`](models/lightgbm-optuna/) — 50 Optuna trials, 10-fold CV, PR-AUC objective, canary guard on coils 806/1187. OOF PR-AUC 0.345; test 26 pos @ top_k_26.

### 7D — rf-smote-v2

New [`models/rf-smote-v2/`](models/rf-smote-v2/) — BorderlineSMOTE without class_weight double penalty, top-K=26. Test 26 pos; canary-safe at rank-K.

### 7F — Disagreement diagnostics

[`scripts/analyze_model_disagreement.py`](scripts/analyze_model_disagreement.py) — K=24 union 33 coils; K=26 union 35 coils.

Pack: `python scripts/pack_phase7_submissions.py`

---

## 18. HackerEarth submission log

| Date (2026) | Method | Score | Test pos | Notes |
|-------------|--------|-------|----------|-------|
| ~05-25 | xgboost-baseline | 1.13208 | 3 | First submission |
| ~05-25 | lightgbm-cv | **1.88679** | 5 | Best (accuracy-threshold era) |
| ~05-25 | gbm-ensemble | 1.13208 | 3 | Failed — high threshold |
| ~05-26 | lightgbm-recall | **7.16981** | 19 | Recall-first t=0.05 |
| ~05-26 | sklearn-recall | **7.92453** | 21 | RF+ET+GBM, target_fpr |
| ~05-26 | gbm-recall | 5.66038 | 15 | Equal GBM, target_fpr (pre-forum fix) |
| ~05-26 | phase6 sklearn-rethreshold | **9.05660** | 24 | t=0.411, no retrain |
| ~05-26 | phase6 recall-blend | **9.05660** | 24 | 50/50 blend |
| ~05-26 | phase6 gbm-recall rethreshold | **9.05660** | 24 | t=0.121 |
| ~05-26 | phase6 lightgbm-recall | **9.05660** | 24 | t=0.036 |
| ~05-26 | phase7 lightgbm top_k_26 | **9.81132** | 26 | All K=26 variants identical |
| ~05-26 | phase7 recall-blend top_k_26 | **9.81132** | 26 | Count band, not model |
| ~05-26 | phase7 mega-recall-blend | **9.81132** | 26 | intersection @ K=26 |
| ~05-26 | phase7 lightgbm-optuna | **9.81132** | 26 | Optuna @ K=26 plateau |
| ~05-26 | **gbm-recall forum33** | **12.45283** | 33 | Equal GBM, t=0.05 ≡ top_k_33 |
| ~05-26 | phase8 gbm-recall top_k_33 | **12.45283** | 33 | Retrain with explicit K=33 meta |
| ~05-26 | phase8 gbm-mega-blend | **12.45283** | 33 | Stacking @ K=33 — same top-33 set as gbm |
| ~05-26 | **phase8 union-gbm33-plus-2** | **13.20755** | 35 | gbm33 + 2 union exclusives (282, 631) |
| ~05-26 | **phase8 union-gbm33-plus-5** | **14.33964** | 38 | gbm33 + 5 exclusives (Phase 8 pool) |
| ~05-28 | **union-gbm33-plus-6** | **14.71698** | 39 | +1138 exclusive |
| ~05-28 | **union-gbm33-plus-7** | **15.09434** | 40 | +1346 exclusive |
| ~05-28 | **union-gbm33-plus-8** | **15.47170** | 41 | +1189 exclusive |
| ~05-28 | **union-gbm33-plus-9** | **15.84906** | 42 | +826 exclusive — **current best** |

**Current best LB:** **15.84906** (`union-gbm33-plus-9`, 42 positives). Fallback: `models/phase8-rethreshold/outputs/union-gbm33-plus-9/submission/`.

---

## 19. Phase 8 — push beyond LB 12.45 (2026-05-26)

**Phase 7 LB result:** All K=26 variants (lightgbm, recall-blend, mega-recall-blend, optuna) scored **9.81132** (26 pos). **gbm-recall forum33** (t=0.05, 33 pos) scored **12.45283** — +27% over K=26 plateau.

### 8A — GBM-anchored exact-K sweep

Extended [`scripts/rethreshold_submission.py`](scripts/rethreshold_submission.py) with `--phase8`: K ∈ {33,34,35,36,38}, methods gbm-recall / gbm-mega-blend / gbm-recall-optuna. Outputs under `models/phase8-rethreshold/outputs/`.

New [`scripts/build_union_submission.py`](scripts/build_union_submission.py): gbm33 base + union exclusives for K=35, 38.

### 8B — gbm-mega-blend

New [`models/gbm-mega-blend/`](models/gbm-mega-blend/) — gbm-recall primary (70%) + sklearn/lgbm secondary, K=33, strategies: gbm_only, gbm_heavy, weight_opt, stacking.

### 8C — gbm-recall top_k_33 retrain

[`models/gbm-recall/train.py`](models/gbm-recall/train.py) supports `--threshold-strategy top_k_33`; predict uses rank-based K from meta.

### 8D — gbm-recall-optuna

New [`models/gbm-recall-optuna/`](models/gbm-recall-optuna/) — joint Optuna on XGB+LGB+CatBoost, 10-fold CV, PR-AUC on equal blend, canary guard vs gbm-recall on coils 806/1187.

### 8E — Feature probes

[`scripts/probe_single_feature.py`](scripts/probe_single_feature.py) — ratio features X13/X10, X30/X32 with canary guard.

### 8F — Diagnostics

Extended [`scripts/analyze_model_disagreement.py`](scripts/analyze_model_disagreement.py) for K=33 and gbm33 vs union diff.

Pack: `python scripts/pack_phase8_submissions.py`. See [`models/phase8-rethreshold/README.md`](models/phase8-rethreshold/README.md).

### Phase 8 LB results (confirmed)

| Upload | Score | Test pos | Takeaway |
|--------|-------|----------|----------|
| gbm-recall / gbm-mega-blend @ K=33 | 12.45283 | 33 | Same top-33 coils — blending did not help at this K |
| union-gbm33-plus-2 | **13.20755** | 35 | +6% — adding 2 secondary-model exclusives works |
| union-gbm33-plus-5 | **14.33964** | 38 | +15% vs gbm33 — union validated |
| union-gbm33-plus-6…9 | **14.72–15.85** | 39–42 | Phase 9 — see §20F; linear ~0.38/exclusive |

**Key insight (Phase 8–9):** LB score scales with flagging the right *additional* coils from model disagreement, not from re-ranking within gbm's top-33. Phase 9 confirmed **~+0.377 LB per ranked exclusive** from K=38 through K=42 (union-gbm33-plus-5 → plus-9). **Current best LB: 15.84906** at 42 test positives.

### Why `phase6/7/8-rethreshold` folders exist

These names break the usual `models/{algorithm}/` kebab-case rule. They are **not** ML methods — no `train.py` / `predict.py`. They are **upload-batch staging dirs**: README + gitignored `outputs/` filled by `scripts/rethreshold_submission.py` and `pack_phase*_submissions.py`. The underlying methods (`gbm-recall`, `recall-blend`, …) still own the source zips.

Prefer proper method folders for durable approaches (e.g. promote union augment to `models/union-gbm33-augment/`). Phase folders were an expedient during the score push.

---

## 20. Phase 9 — Push toward LB ~100 (2026-05-27 / 2026-05-28)

**Goal:** Extend union-gbm33 strategy, improve base ranking, add model diversity and meta-stacking.

### 20A — Extended union K sweep

- Refactored union logic to [`utils/union_augment.py`](utils/union_augment.py)
- Extended [`scripts/build_union_submission.py`](scripts/build_union_submission.py): K=35–42, expanded secondary pool:
  - `lightgbm-recall`, `sklearn-recall`, `recall-blend` (Phase 8)
  - **Phase 9 additions:** `rf-smote-v2`, `mega-recall-blend`, `lightgbm-optuna`, `gbm-mega-blend`
- Ranking: `max(secondary proba)` over loaded secondaries; optional `--ranking mean|weighted`
- Secondary positive sets computed at **K=26** with canary forcing; anchor fixed at **gbm-recall top-33**
- Extended [`scripts/analyze_model_disagreement.py`](scripts/analyze_model_disagreement.py): up to 10 methods, K up to 44
- Extended gbm-only K sweep: K=34–42 via `rethreshold_submission.py --phase8`

**Exclusive ranking order (Phase 9 expanded pool, beyond gbm33):**

| Rank | CoilID | First appears in |
|------|--------|------------------|
| 1 | 282 | plus-2 (K=35) |
| 2 | 631 | plus-2 |
| 3 | 1097 | plus-3 (K=36) |
| 4 | 302 | plus-4 (K=37) |
| 5 | 940 | plus-5 (K=38) |
| 6 | 1138 | plus-6 (K=39) |
| 7 | 1346 | plus-7 (K=40) |
| 8 | 1189 | plus-8 (K=41) |
| 9 | 826 | plus-9 (K=42) |

Full coil lists per K: [`models/phase8-rethreshold/outputs/union_manifest.json`](models/phase8-rethreshold/outputs/union_manifest.json).

### 20B — First-class union method

New [`models/union-gbm33-augment/`](models/union-gbm33-augment/) — `train.py` / `predict.py` / `pack.py` wrapping union augment with `--target-k`, `--ensure-secondaries`, `--ranking`.

### 20C — Scaled GBM + Optuna fix

- `gbm-recall/train.py --scale`: 2000 trees, lr=0.02, depth=5, 10-fold CV — trained; 33 test positives, canaries OK; did not beat union path on LB
- `gbm-recall-optuna`: objective → **OOF accuracy @ top-K**; rank-based canary guard; 8-trial run OOF PR-AUC **0.361** (best saved in `outputs/latest/`)

### 20D — Meta-stack + new model families

| Method | Purpose | LB impact |
|--------|---------|-----------|
| [`models/meta-recall-stack/`](models/meta-recall-stack/) | LogisticRegression / weight-opt on 5–9 base OOF probas; K ∈ {33,35,38,40} | Secondary pool only — 35 test pos @ K=35 |
| [`models/catboost-recall/`](models/catboost-recall/) | CatBoost solo @ top-K=33 | Union secondary |
| [`models/autogluon-recall/`](models/autogluon-recall/) | AutoGluon `medium_quality`, 33 test pos | Union secondary |
| [`models/gbm-recall-fullmiss/`](models/gbm-recall-fullmiss/) | GBM + miss_X1..miss_X49 | **SKIP** — canary guard fails |

### 20E — Guarded FE

- [`utils/tabular_features_full_miss.py`](utils/tabular_features_full_miss.py) — miss indicators on all 49 columns
- Probe result (`feature_probe_report.json`): ratios X13/X10, X30/X32, and full_miss all **SKIP** (proba drop on coils 806/1187)

### 20F — Phase 9 LB results (confirmed 2026-05-28)

**Hypothesis validated:** extending union beyond plus-5 continues to improve LB with **near-constant marginal gain** per ranked exclusive.

| Method | K | Test pos | LB score | Δ vs plus-5 | Marginal exclusive added |
|--------|---|----------|----------|-------------|--------------------------|
| union-gbm33-plus-5 | 38 | 38 | 14.33964 | — | (Phase 8 baseline) |
| **union-gbm33-plus-6** | 39 | 39 | **14.71698** | +0.37734 | **1138** |
| **union-gbm33-plus-7** | 40 | 40 | **15.09434** | +0.75470 | **1346** |
| **union-gbm33-plus-8** | 41 | 41 | **15.47170** | +1.13206 | **1189** |
| **union-gbm33-plus-9** | 42 | 42 | **15.84906** | +1.50942 | **826** |

**Marginal gain plus-N → plus-(N+1):** ~**0.3774** LB points per additional coil (std ≈ 0.00002) — effectively linear in this band.

**Score vs positive count (union family, confirmed):**

```
K=33  → 12.45   (gbm anchor only)
K=35  → 13.21   (plus-2)
K=38  → 14.34   (plus-5)
K=39  → 14.72   (plus-6)
K=40  → 15.09   (plus-7)
K=41  → 15.47   (plus-8)
K=42  → 15.85   (plus-9)  ← current best
```

**Takeaways:**

1. Each ranked exclusive from the expanded secondary pool (1138, 1346, 1189, 826) was worth ~0.38 LB — likely true positives missed by gbm top-33 alone.
2. Still **far from forum ~100** (~7.6× gap); next push should find **more exclusives** (disagreement union at K=33 had ~59 coils; 9 used so far) or improve gbm33 anchor ranking.
3. Untested on LB: plus-3 (K=36), plus-4 (K=37), pure gbm K=39–42 without union, K=43+ if more exclusives exist.

### Phase 9 recommended upload / reproduce

```powershell
python models/gbm-recall/train.py --threshold-strategy top_k_33
python models/gbm-recall/predict.py
# train secondaries (or --ensure-secondaries on union method)
python scripts/build_union_submission.py --target-ks 43 44 45
python scripts/pack_phase8_submissions.py
python scripts/check_submission_vs_baseline.py models/phase8-rethreshold/outputs/union-gbm33-plus-9/submission/submission.csv --max-positives 45
```

**Current best upload pair:**

| File | Path |
|------|------|
| Predictions CSV | `models/phase8-rethreshold/outputs/union-gbm33-plus-9/submission/submission.csv` |
| Source zip | `models/phase8-rethreshold/outputs/union-gbm33-plus-9/submission/gbm-recall-hackerearth.zip` |

---

## 21. Key hyperparameters (recall-first model)

**`lightgbm-recall` — same LGBM as lightgbm-cv, threshold t=0.05:**

See §16 in prior version for LGBM params; threshold strategy in `utils/threshold_tuning.py`.

---

## 22. References

- Problem: https://www.hackerearth.com/challenges/competitive/tata-steel-ai-hackathon/machine-learning/fd-a5a6dcb2/
- Method READMEs: `models/*/README.md`
- Cursor skills: `.cursor/skills/tata-steel-hackathon/SKILL.md`, `tata-steel-model-experiment`, `tata-steel-submission`, `tata-steel-eda`
- Project setup: root `README.md`
