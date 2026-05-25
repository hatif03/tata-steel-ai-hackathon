# HackerEarth upload bundle

Problem: [ML challenge](https://www.hackerearth.com/challenges/competitive/tata-steel-ai-hackathon/machine-learning/fd-a5a6dcb2/)

Upload **two separate files** to HackerEarth:

| File | Description |
|------|-------------|
| `submission.csv` | Predictions — columns `CoilID,Y`, 339 rows, values 0 or 1 |
| `xgboost-baseline-hackerearth.zip` | Source archive with `approach.txt` + `source/` |

## If files are missing here

`submission.csv`, `source/`, the zip, and `pack_meta.json` are **generated locally** and listed in `.gitignore` (not stored in git). After clone or cleanup, regenerate:

```powershell
.\.venv\Scripts\Activate.ps1
python models/xgboost-baseline/train.py    # if no outputs/ yet
python models/xgboost-baseline/predict.py
python models/xgboost-baseline/pack.py     # restores everything in this folder
```

You should then see:

```
submission/
  approach.txt                      # in git — edit before upload
  README.md                         # in git
  submission.csv                    # generated
  xgboost-baseline-hackerearth.zip  # generated
  pack_meta.json                    # generated
  source/                           # generated
```

## Regenerate after train/predict changes

```powershell
python models/xgboost-baseline/predict.py
python models/xgboost-baseline/pack.py
```

Or from repo root:

```powershell
python .cursor/skills/tata-steel-submission/scripts/pack_submission.py models/xgboost-baseline
```

## Contents

- `approach.txt` — approach, feature engineering, tools, reproduction steps (edit before each upload)
- `submission.csv` — copied from latest run (auto-updated by pack script; **gitignored**)
- `source/` — snapshot of notebook, train.py, predict.py, README, requirements, utils (auto-generated; **gitignored**)
- `pack_meta.json` — audit trail (auto-generated; **gitignored**)
- `*-hackerearth.zip` — upload archive (auto-generated; **gitignored**)

Do not edit files in `source/` by hand — they are refreshed on each pack run.
Only `approach.txt` and this README are tracked in git under `submission/`.
