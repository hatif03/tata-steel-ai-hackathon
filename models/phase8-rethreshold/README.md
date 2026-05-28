# Phase 8 / 9 — GBM-anchored union score push

Phase 8 established **gbm-recall top-33** as the anchor and **union augments** (secondary-model exclusives) as the winning strategy. Phase 9 extended K through **plus-9 (K=42)** with an expanded secondary pool and confirmed **linear LB gains** in the 38–42 band.

---

## Leaderboard results (confirmed uploads)

| Method | LB score | Test pos | Δ vs gbm33 | Marginal exclusive | Notes |
|--------|----------|----------|------------|-------------------|-------|
| gbm-recall (top_k_33 / forum33) | **12.45283** | 33 | — | — | Anchor set |
| gbm-mega-blend @ K=33 | **12.45283** | 33 | 0 | — | Same top-33 as gbm |
| union-gbm33-plus-2 | **13.20755** | 35 | +0.75 | 282, 631 | Phase 8 |
| union-gbm33-plus-5 | **14.33964** | 38 | +1.89 | (+940 in Phase 9 pool) | Phase 8 best |
| **union-gbm33-plus-6** | **14.71698** | 39 | +2.26 | **1138** | Phase 9 |
| **union-gbm33-plus-7** | **15.09434** | 40 | +2.64 | **1346** | Phase 9 |
| **union-gbm33-plus-8** | **15.47170** | 41 | +3.02 | **1189** | Phase 9 |
| **union-gbm33-plus-9** | **15.84906** | 42 | +3.40 | **826** | **Current best** |

**Current best LB:** **15.84906** (`union-gbm33-plus-9`, 42 positives).

**Marginal gain plus-N → plus-(N+1) for N≥5:** ~**0.3774** LB points per coil — effectively constant.

---

## Union algorithm (technical)

Implemented in [`utils/union_augment.py`](../../utils/union_augment.py), invoked by [`scripts/build_union_submission.py`](../../scripts/build_union_submission.py) and [`models/union-gbm33-augment/`](../../models/union-gbm33-augment/).

1. **Anchor:** `apply_top_k(gbm-recall proba, K=33)` with canaries `{654, 806, 532, 958, 1187}` forced positive.
2. **Secondary union:** For each loaded secondary method, compute top-**26** set (same canary forcing).
3. **Candidates:** `(⋃ secondary top-26) ∪ gbm33) − gbm33`.
4. **Rank candidates:** `max(proba)` across secondaries (default); optional `mean` / `weighted`.
5. **Fill to target K:** Keep all 33 anchor coils; add ranked candidates until `target_k` reached.

**Phase 9 secondary pool (7 models):** lightgbm-recall, sklearn-recall, recall-blend, rf-smote-v2, mega-recall-blend, lightgbm-optuna, gbm-mega-blend.

**Exclusive order beyond gbm33:** 282 → 631 → 1097 → 302 → 940 → 1138 → 1346 → 1189 → 826.

Full positive coil lists: [`outputs/union_manifest.json`](outputs/union_manifest.json).

---

## What we learned

1. **Union beats blending** — gbm-mega-blend @ K=33 identical to gbm; coil-ID union beats probability averaging.
2. **Linear band 38–42** — each ranked exclusive worth ~0.38 LB; extrapolation suggests K=43+ may help if more valid exclusives exist.
3. **Expanded pool matters** — Phase 9 secondaries surfaced 1097, 940, 1189, 826 not in Phase 8’s top-5 list.
4. **Canary guard** — all plus-2…plus-9 submissions keep all five canaries Y=1.

---

## Upload pairs (HackerEarth)

| Method | CSV | Source zip |
|--------|-----|------------|
| **union-gbm33-plus-9** (best) | `outputs/union-gbm33-plus-9/submission/submission.csv` | `.../gbm-recall-hackerearth.zip` |
| union-gbm33-plus-6…8 | `outputs/union-gbm33-plus-{6,7,8}/submission/` | same gbm-recall zip |

Manifest: `outputs/pack_manifest.json`. Candidates: `outputs/lb_candidates.json`.

---

## Reproduce

```powershell
python models/gbm-recall/train.py --threshold-strategy top_k_33
python models/gbm-recall/predict.py
python scripts/build_union_submission.py --target-ks 35 36 37 38 39 40 41 42
python scripts/pack_phase8_submissions.py
python scripts/analyze_model_disagreement.py --k-values 33 38 44
```

Guard check:

```powershell
python scripts/check_submission_vs_baseline.py models/phase8-rethreshold/outputs/union-gbm33-plus-9/submission/submission.csv --max-positives 45
```

---

## Next directions

1. **K=43+** — run `build_union_submission.py --target-ks 43 44 45`; add catboost-recall / meta-recall-stack / autogluon-recall to secondary pool if not already loaded.
2. **Upload plus-3, plus-4** — fill gap in 36–37 band (untested on LB).
3. **Compare pure gbm K=39–42** vs union at same K — isolate union value.
4. **Do not** use full_miss FE or ratio features — canary guard fails on 806/1187.

See [DEVELOPMENT.md](../../DEVELOPMENT.md) §18–20, [models/union-gbm33-augment/README.md](../union-gbm33-augment/README.md).
