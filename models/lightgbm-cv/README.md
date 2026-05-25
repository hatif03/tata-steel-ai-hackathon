# LightGBM CV

LightGBM classifier with **native missing-value handling** and **missingness indicators**, improving on the XGBoost baseline for the [Tata Steel AI Hackathon](https://www.hackerearth.com/challenges/competitive/tata-steel-ai-hackathon/) Round 1 ML challenge.

## Method

- **Algorithm**: `LGBMClassifier` (LightGBM gradient boosted trees)
- **Libraries**: `lightgbm`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- **Hyperparameters**:
  - `n_estimators=500`, `max_depth=4`, `learning_rate=0.05`
  - `subsample=0.8`, `colsample_bytree=0.8`
  - `scale_pos_weight = n_neg / n_pos` ≈ 19.48
  - `objective='binary'`, `random_state=42`, `n_jobs=-1`
- **Feature engineering** (`features.py`):
  - Raw `X1`–`X49` with **NaN retained** (LightGBM native missing splits)
  - Binary `miss_*` indicators for columns with meaningful train missing rates
  - `row_missing_count` — total missing features per row
- **Preprocessing**: no imputation; missingness encoded explicitly

## Why this approach

The XGBoost baseline median-imputed all nulls before training, discarding LightGBM/XGBoost-style missing split direction. **X15 alone is null in ~12% of rows**, and positive coils have slightly higher missing-row rates. LightGBM routes missing values to dedicated tree branches; adding explicit missing indicators gives the model both implicit and explicit missingness signal.

**Strengths**: better OOF PR-AUC and positive recall than baseline; still fast on 1,352 rows; no external data.

**Trade-offs**: fold PR-AUC still varies (0.20–0.60) on small data; threshold tuning remains sensitive; no ratio/interaction features yet.

## Technical details

### Preprocessing

1. Drop `CoilID` from features.
2. Build 60-column matrix: 49 raw features + 10 missing indicators + `row_missing_count`.
3. Pass DataFrame directly to LightGBM (preserves feature names).

### Train / validation

- **Stratified 5-fold CV** (`StratifiedKFold`, `shuffle=True`, `random_state=42`).
- OOF positive-class probabilities; threshold sweep on OOF to maximize accuracy.

### Class imbalance

\[
\text{scale\_pos\_weight} = \frac{n_0}{n_1} \approx 19.48
\]

Monitor PR-AUC; optimize accuracy via threshold \(t^*\) on OOF probabilities.

### Leakage guards

- No `CoilID` in features.
- No test labels used for tuning.
- Missing indicators computed per row from raw values only (no target encoding).

### Output layout

Same structure as other methods under `outputs/runs/<timestamp>/` — see `utils/run_artifacts.py`.

## Results

| Metric | lightgbm-cv | xgboost-baseline | Δ |
|--------|-------------|------------------|---|
| OOF PR-AUC | **0.3541** | 0.3269 | +0.027 |
| OOF accuracy @ tuned threshold | **95.86%** (t=0.730) | 95.56% (t=0.810) | +0.30 pp |
| Positive recall (OOF) | **21.2%** | 13.6% | +7.6 pp |
| OOF positives predicted | 18 | 12 | +6 |
| Test positives predicted | **5** | 3 | +2 |
| Majority baseline | 95.12% | 95.12% | — |
| Leaderboard (HackerEarth) | _After submit_ | 1.13208 | — |

Fold PR-AUC: 0.60, 0.20, 0.25, 0.48, 0.58.

See `outputs/latest/metrics.json` after each run.

## How to run

### Local

```powershell
.\.venv\Scripts\Activate.ps1
python models/lightgbm-cv/train.py
python models/lightgbm-cv/predict.py
python models/lightgbm-cv/pack.py
```

### HackerEarth upload

Upload from `models/lightgbm-cv/submission/`:

- `submission.csv` — predictions
- `lightgbm-cv-hackerearth.zip` — source + approach.txt

### Google Colab

Open `models/lightgbm-cv/lightgbm-cv.ipynb`, set `REPO_ROOT` if needed, run all cells.
