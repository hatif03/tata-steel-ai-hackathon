# Tata Steel AI Hackathon

Tabular ML challenge: predict binary target `Y` from coil features `X1`–`X49`.

## Setup (virtual environment)

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

Dependencies are listed in `requirements.txt`.

## Train a model

```powershell
python models/xgboost-baseline/train.py
python models/xgboost-baseline/predict.py
python models/xgboost-baseline/pack.py
```

Outputs land under `models/xgboost-baseline/outputs/`:

| Path | Contents |
|------|----------|
| `outputs/latest/artifacts/` | **Saved models** (copied from latest run) |
| `outputs/latest/plots/` | Metric plots |
| `outputs/latest/predictions/submission.csv` | HackerEarth upload |
| `outputs/runs/<timestamp>/` | Full timestamped run archive |

### Saved model files (per run)

- `xgb_model.json` — native XGBoost model (~660 KB)
- `model.joblib` — sklearn `XGBClassifier` wrapper (reload with joblib)
- `imputer.joblib` — fitted median imputer
- `meta.joblib` — threshold, features, hyperparameters
- `manifest.json` — file listing with sizes

## HackerEarth submission

After train + predict, package both required uploads:

```powershell
python .cursor/skills/tata-steel-submission/scripts/pack_submission.py models/xgboost-baseline
```

| Upload to HackerEarth | Path |
|-----------------------|------|
| Predictions (CSV) | `models/{method}/submission/submission.csv` |
| Source code (zip/tar) | `models/{method}/submission/{method}-hackerearth.zip` |

Edit `models/{method}/submission/approach.txt` before packing (approach, feature engineering, tools).

Validate only:

```powershell
python .cursor/skills/tata-steel-submission/scripts/validate_submission.py models/xgboost-baseline/submission/submission.csv
```

## Project layout

```
dataset/                  # train.csv, test.csv (committed)
models/{method}/          # one folder per ML approach (train.py, predict.py, …)
  submission/
    approach.txt          # committed — edit before upload
    submission.csv        # generated (gitignored) — run pack.py
    *-hackerearth.zip     # generated (gitignored) — run pack.py
  outputs/                # generated (gitignored) — run train.py
utils/                    # shared helpers (committed)
scripts/                  # cross-method tools (rethreshold, pack batches, EDA)
scripts/setup.ps1         # create .venv (gitignored)
.cursor/skills/           # Cursor agent skills (committed)
```

**Exception — `models/phase*-rethreshold/`:** Not full methods. Upload-batch staging only (README + gitignored `outputs/`). See [`models/phase8-rethreshold/README.md`](models/phase8-rethreshold/README.md#why-is-this-folder-called-phase8-rethreshold).

See `models/xgboost-baseline/README.md` for method details.

## Development history

Full log of experiments, leaderboard scores, failures, and lessons learned:

**[DEVELOPMENT.md](DEVELOPMENT.md)** — chronology, metrics, recall-first evolution (Phase 5), upload order.

### Current best submission

| LB score | Method | Test positives | Path |
|----------|--------|----------------|------|
| **23.01887** | vote-union-m2-k40 | 61 | `models/phase8-rethreshold/outputs/vote-union-m2-k40/submission/` |
| 18.49057 | union-gbm33-plus-16 | 49 | `models/phase8-rethreshold/outputs/union-gbm33-plus-16/submission/` |

Phase 11 **vote union** at K=40, M≥2 confirmed **+4.5 LB** over union augment (18.49 @ 49 pos). Marginal ~**0.377 LB/point** holds; path to LB 100 needs precision filters, not count alone. See [DEVELOPMENT.md](DEVELOPMENT.md) §22.

**Phase 11 confirmed results:**

| Method | LB | Positives | Notes |
|--------|-----|-----------|-------|
| **vote-union-m2-k40** | **23.01887** | 61 | **Current best** |
| vote-union-m3-k40 | 19.24529 | 51 | Stricter M hurts |
| union-gbm33-plus-16 | 18.49057 | 49 | Phase 10 best |
| vote-union-m2-k33 | 17.73585 | 47 | K too low |

Durable method folders: [`models/vote-union-recall/`](models/vote-union-recall/), [`models/union-gbm33-augment/`](models/union-gbm33-augment/). Full score history: [DEVELOPMENT.md](DEVELOPMENT.md) §18, §21–22.

```powershell
python scripts/build_vote_union.py
python scripts/pack_phase8_submissions.py
python scripts/check_submission_vs_baseline.py models/phase8-rethreshold/outputs/vote-union-m2-k40/submission/submission.csv --max-positives 80
```

## What gets committed vs generated

| Path | Git | Regenerate |
|------|-----|------------|
| `models/*/submission/approach.txt` | Yes | Edit manually |
| `models/*/outputs/` | No | `train.py` |
| `models/*/submission/submission.csv` | No | `pack.py` |
| `models/*/submission/*-hackerearth.zip` | No | `pack.py` |
