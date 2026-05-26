# Phase 7 exact-K rethreshold uploads

Rank-based **top-K** selection with canary coils forced positive. Builds on LB **9.05660** (Phase 6 @24 pos).

## Generate submissions

```powershell
python scripts/rethreshold_submission.py --write-all --write-best --pack --guard-check
python scripts/pack_phase7_submissions.py
python scripts/analyze_model_disagreement.py
```

## Top upload candidates (see `outputs/lb_candidates.json`)

| Method | K | OOF acc | CSV |
|--------|---|---------|-----|
| lightgbm-recall | 26 | 0.957 | `outputs/lightgbm-recall/submission/submission.csv` |
| recall-blend | 25 | 0.958 | `outputs/recall-blend/submission_k25.csv` |
| mega-recall-blend | 26 | 0.956 | `outputs/mega-recall-blend/submission/submission.csv` |
| gbm-recall forum33 | 33 | — | `outputs/gbm-recall-forum33/submission/submission.csv` |

Each folder under `outputs/{method}/submission/` has CSV + `*-hackerearth.zip`.

**Note:** Zips are gitignored; regenerate locally after clone.
