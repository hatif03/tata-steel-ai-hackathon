# Phase 6 re-threshold uploads

Re-thresholded predictions (24–28 test positives) without retraining.

## Pack zips (HackerEarth source bundle)

```powershell
python scripts/pack_phase6_submissions.py
```

Upload pairs (prediction CSV + source zip):

| Method | CSV | ZIP |
|--------|-----|-----|
| sklearn-recall | `outputs/sklearn-recall/submission/submission.csv` | `outputs/sklearn-recall/submission/sklearn-recall-hackerearth.zip` |
| recall-blend | `outputs/recall-blend/submission/submission.csv` | `outputs/recall-blend/submission/recall-blend-hackerearth.zip` |
| gbm-recall | `outputs/gbm-recall/submission/submission.csv` | `outputs/gbm-recall/submission/gbm-recall-hackerearth.zip` |
| lightgbm-recall | `outputs/lightgbm-recall/submission/submission.csv` | `outputs/lightgbm-recall/submission/lightgbm-recall-hackerearth.zip` |

See `outputs/lb_candidates.json` and `outputs/pack_manifest.json`.

**Note:** `*-hackerearth.zip` files are listed in `.gitignore` (regenerate locally after clone). They exist on disk after running the pack script.
