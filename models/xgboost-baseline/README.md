# XGBoost Baseline

First strong baseline for the [Tata Steel AI Hackathon](https://www.hackerearth.com/community/challenges/competitive/tata-steel-ai-hackathon/) Round 1 ML challenge.

## Method

- **Algorithm**: `XGBClassifier` (gradient boosted decision trees)
- **Libraries**: `xgboost`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- **Hyperparameters** (baseline defaults):
  - `n_estimators=500`, `max_depth=4`, `learning_rate=0.05`
  - `subsample=0.8`, `colsample_bytree=0.8`
  - `scale_pos_weight = n_neg / n_pos` (handles ~4.9% positive rate)
  - `objective='binary:logistic'`, `eval_metric='logloss'`
  - `random_state=42`, `n_jobs=-1`
- **Preprocessing**: median imputation per CV fold (`SimpleImputer`); no feature scaling (tree model)

## Why this approach

Steel-process tabular data with mixed continuous and count-like features (X1–X49) is a strong fit for gradient boosting: nonlinear interactions, robustness to feature scale, and native handling of missing values when passed to XGBoost directly. This baseline establishes a reproducible floor before tuning (LightGBM/CatBoost) or ensembling.

**Strengths**: fast to train on 1,352 rows; interpretable feature importances; `scale_pos_weight` addresses class imbalance without synthetic oversampling (avoids SMOTE leakage risk on small data).

**Trade-offs**: no feature engineering yet; threshold tuned on OOF probabilities for accuracy, not default 0.5; single model vs ensemble.

## Technical details

### Preprocessing

1. Drop `CoilID` from features (identifier only).
2. Median impute inside each CV fold (fit on train fold, transform val fold) to prevent leakage.
3. Final model: imputer fit on full train, transform test.

### Train / validation

- **Stratified 5-fold CV** (`StratifiedKFold`, `shuffle=True`, `random_state=42`) preserves ~4.9% positive rate per fold.
- Out-of-fold (OOF) positive-class probabilities collected for all train rows.

### Class imbalance

Positive rate \(p \approx 0.049\). Set:

\[
\text{scale\_pos\_weight} = \frac{n_0}{n_1} = \frac{1286}{66} \approx 19.48
\]

Monitor **PR-AUC** on OOF probs; optimize **accuracy** via threshold sweep on OOF:

\[
\hat{y} = \mathbb{1}[\hat{p} \geq t^*], \quad t^* = \arg\max_t \text{Accuracy}(y, \hat{p} \geq t)
\]

### Output layout (local & notebook runs)

Each run is saved under `outputs/runs/<timestamp>/`:

```
models/xgboost-baseline/outputs/
  latest_run.txt              # pointer to most recent run id
  latest/                     # copied summary (metrics, plots, models, submission)
  runs/<timestamp>/
    metrics.json
    run_config.json
    oof_predictions.csv
    artifacts/
      xgb_model.json          # native XGBoost (~660 KB)
      model.joblib            # sklearn wrapper
      imputer.joblib
      meta.joblib
      manifest.json
    plots/
    predictions/
      submission.csv
      test_predictions.csv
```

Shared helpers live in `utils/run_artifacts.py` and `utils/plotting.py`.

### Data leakage guards

- No use of `CoilID` as a feature.
- Imputer statistics computed only on training fold (CV) or full train (inference).
- No external data; no target encoding without nested CV.
- Test labels never used for tuning.

### Inference pipeline

1. Fit `SimpleImputer(median)` on full `train[features]`.
2. Fit `XGBClassifier` on imputed full train.
3. `predict_proba(test)[:, 1]` → apply `t*` → integer `{0,1}` predictions.
4. Write `predictions/submission.csv` with columns `CoilID,Y`.

## Results

| Metric | Value |
|--------|-------|
| OOF PR-AUC | 0.3269 |
| OOF accuracy @ tuned threshold (t=0.810) | 95.56% |
| Majority baseline accuracy | 95.12% |
| Leaderboard (HackerEarth) | _After submit_ |

See `outputs/latest/metrics.json` after each local run.

## How to run

### Local (recommended for training on your machine)

```powershell
# One-time setup
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
.\.venv\Scripts\Activate.ps1

# Train: CV, metrics, plots, saved models → outputs/runs/<timestamp>/
python models/xgboost-baseline/train.py

# Predict using latest run → predictions/submission.csv
python models/xgboost-baseline/predict.py
```

### HackerEarth submission folder

After train + predict, package uploads under `submission/`:

```
models/xgboost-baseline/submission/
  approach.txt                    # approach, feature engineering, tools
  submission.csv                  # prediction file (CoilID,Y)
  xgboost-baseline-hackerearth.zip  # source archive for HackerEarth
```

```powershell
python models/xgboost-baseline/pack.py
```

Or:

```powershell
python .cursor/skills/tata-steel-submission/scripts/pack_submission.py models/xgboost-baseline
```

Upload **submission.csv** (predictions) and **xgboost-baseline-hackerearth.zip** (source) to
[the problem page](https://www.hackerearth.com/challenges/competitive/tata-steel-ai-hackathon/machine-learning/fd-a5a6dcb2/).

Optional flags:

```bash
python models/xgboost-baseline/train.py --run-id my-experiment
python models/xgboost-baseline/predict.py --run-dir models/xgboost-baseline/outputs/runs/my-experiment
```

### Google Colab

1. Open `models/xgboost-baseline/xgboost-baseline.ipynb`.
2. Set `REPO_ROOT` / `DATA_DIR` in the paths cell if not using the default Drive layout.
3. Run all cells — outputs are written into the same `outputs/runs/` structure under the repo clone on Drive.
4. Run `pack_submission.py` and upload `submission/submission.csv` + the zip archive.
