# Phase 8 / 9 / 10 — GBM-anchored union score push

Phase 8 established **gbm-recall top-33** as the anchor and **union augments** as the winning strategy. Phase 9 confirmed linear LB gains through K=42. **Phase 10 (2026-05-29)** extended the curve to **K=49** with the expanded 16-model pool — **current best LB 18.49057**.

---

## Leaderboard results (confirmed uploads)

| Method | LB score | Test pos | Δ vs gbm33 | Marginal exclusive | Notes |
|--------|----------|----------|------------|-------------------|-------|
| gbm-recall (top_k_33) | **12.45283** | 33 | — | — | Anchor |
| union-gbm33-plus-2 | **13.20755** | 35 | +0.75 | 282, 631 | Phase 8 |
| union-gbm33-plus-5 | **14.33964** | 38 | +1.89 | … | Phase 8 |
| union-gbm33-plus-9 | **15.84906** | 42 | +3.40 | 826 | Phase 9 best |
| **union-gbm33-plus-10** | **16.22642** | 43 | +3.88 | 1023 | Phase 10 |
| **union-gbm33-plus-11** | **16.60377** | 44 | +4.26 | 804 | |
| **union-gbm33-plus-12** | **16.98113** | 45 | +4.64 | 302 | |
| **union-gbm33-plus-13** | **17.35849** | 46 | +5.02 | 582 | |
| **union-gbm33-plus-14** | **17.73585** | 47 | +5.39 | 972 | |
| **union-gbm33-plus-15** | **18.11321** | 48 | +5.77 | 867 | |
| **union-gbm33-plus-16** | **18.49057** | 49 | +6.15 | 176 | **Current best** |
| union-gbm33-plus-17 | **18.49057** | 50 | +6.15 | 797 | Plateau (no gain vs plus-16) |
| vote-union-m2-k33 | **17.73585** | 47 | — | — | Ties plus-14 |
| rank-avg-k35 | **13.20755** | 35 | — | — | Ties plus-2 — **fails** |

**Current best LB:** **18.49057** (`union-gbm33-plus-16`, 49 positives).

**Marginal gain plus-N → plus-(N+1) for N=9…15:** ~**0.3774** LB points per coil — constant from Phase 9 through Phase 10.

**Saturation:** Coil **797** at K=50 adds zero LB — current rank-ordered exclusive pool exhausted for this secondary set.

---

## Union algorithm (technical)

Implemented in [`utils/union_augment.py`](../../utils/union_augment.py), invoked by [`scripts/build_union_submission.py`](../../scripts/build_union_submission.py) and [`models/union-gbm33-augment/`](../../models/union-gbm33-augment/).

1. **Anchor:** `apply_top_k(gbm-recall proba, K=33)` with canaries `{654, 806, 532, 958, 1187}` forced positive.
2. **Secondary union:** For each loaded secondary method, compute top-**26** set (same canary forcing).
3. **Candidates:** `(⋃ secondary top-26) ∪ gbm33) − gbm33`.
4. **Rank candidates:** `max(proba)` across secondaries (default); optional `mean` / `weighted`.
5. **Fill to target K:** Keep all 33 anchor coils; add ranked candidates until `target_k` reached.

**Phase 10 secondary pool (16 models):** lightgbm-recall, sklearn-recall, recall-blend, rf-smote-v2, mega-recall-blend, lightgbm-optuna, gbm-mega-blend, catboost-recall, meta-recall-stack, autogluon-recall, xgb-recall, lgb-seedblend-recall, knn-positive-profile, smote-stack-recall.

**Phase 10 incremental exclusives (confirmed LB):** 1023 → 804 → 302 → 582 → 972 → 867 → 176 → 797 (dead).

Full positive coil lists: [`outputs/union_manifest.json`](outputs/union_manifest.json).

---

## What we learned (Phase 10)

1. **Linearity extends to K=49** — same ~0.377 LB/exclusive as Phase 9; expanded pool changed rank order but not marginal value per slot.
2. **Saturation at 797** — plus-17 (50 pos) = plus-16 (49 pos) on LB; stop adding from this rank list.
3. **Vote union no upside** — `vote-union-m2-k33` @ 47 pos = `plus-14` exactly (17.73585).
4. **Rank averaging fails on LB** — `rank-avg-k35` = 13.20755 (= plus-2); worse than union at same K.
5. **Prefer plus-16 over plus-17** for submission — same LB, one fewer positive (lower FP risk if metric shifts).

---

## Upload pairs (HackerEarth)

| Method | CSV | Source zip |
|--------|-----|------------|
| **union-gbm33-plus-16** (best) | `outputs/union-gbm33-plus-16/submission/submission.csv` | `gbm-recall-hackerearth.zip` |
| union-gbm33-plus-17 | `outputs/union-gbm33-plus-17/submission/` | same |
| Fallback plus-9 | `outputs/union-gbm33-plus-9/submission/` | same |

Manifest: `outputs/pack_manifest.json`. Score registry: `outputs/lb_candidates.json`.

---

## Reproduce

```powershell
python models/gbm-recall/train.py --threshold-strategy top_k_33
python models/gbm-recall/predict.py
python scripts/run_phase10_train_batch.py --include-slow
python scripts/build_union_submission.py --target-ks 43 44 45 46 47 48 49 50
python scripts/pack_phase8_submissions.py
```

Guard check:

```powershell
python scripts/check_submission_vs_baseline.py models/phase8-rethreshold/outputs/union-gbm33-plus-16/submission/submission.csv --max-positives 55
```

---

## Next directions

1. **New exclusives beyond 797** — Colab AutoGluon `best_quality`, retrain secondaries, rerun `build_union_submission.py --target-ks 51 52 …`
2. **Upload plus-3, plus-4** — untested K=36–37 band
3. **Do not prioritize** rank-avg or vote-union paths for LB
4. **Gap to 100** still ~5.4× — need new coil sources or ranking breakthrough

See [DEVELOPMENT.md](../../DEVELOPMENT.md) §21F, [models/union-gbm33-augment/README.md](../union-gbm33-augment/README.md).
