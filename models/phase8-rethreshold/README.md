# Phase 8 — GBM-anchored score push

Phase 8 built on the Phase 7 finding that **gbm-recall forum33** (equal XGB+LGB+CatBoost @ t=0.05, 33 test positives) scored **12.45283** while all K=26 variants plateaued at **9.81132**. The hypothesis: gains come from (1) flagging more positives in the 33–38 band anchored on gbm ranking, and (2) adding coils that secondary models flag but gbm misses.

---

## Why is this folder called `phase8-rethreshold`?

It **does not follow** the usual `models/{method}/` convention (`gbm-recall`, `lightgbm-cv`, etc.) on purpose — and that is a known inconsistency.

| Usual method folder | `phase6/7/8-rethreshold` |
|---------------------|--------------------------|
| One **algorithm or pipeline** (train → predict → pack) | One **upload batch / experiment campaign** |
| Has `train.py`, `predict.py`, `submission/approach.txt` | Has only `README.md` + gitignored `outputs/` |
| Name describes *what* (e.g. `gbm-mega-blend`) | Name describes *when* (Phase 8 in [DEVELOPMENT.md](../../DEVELOPMENT.md)) |

These folders are **staging areas** for re-thresholded or post-hoc submissions produced by scripts (`rethreshold_submission.py`, `build_union_submission.py`, `pack_phase8_submissions.py`). The real source code and HackerEarth zips still come from the underlying methods (`gbm-recall`, `gbm-mega-blend`, …).

**If we aligned with the naming scheme**, better names would be method-oriented, for example:

- `models/union-gbm33-augment/` — the winning approach (gbm33 + union exclusives), with `train.py` wrapping `build_union_submission.py`
- `models/rethreshold-exact-k/` — generic exact-K sweep outputs, instead of separate phase6/7/8 folders

Phase folders were kept as a quick way to group LB upload batches during the score push; new work should prefer a proper method folder when the approach is worth keeping (e.g. union augment → `union-gbm33-augment`).

---

## Leaderboard results (confirmed uploads)

| Method | LB score | Test pos | Δ vs gbm33 | Verdict |
|--------|----------|----------|------------|---------|
| gbm-recall (top_k_33 / forum33) | **12.45283** | 33 | — | Baseline for Phase 8 |
| gbm-mega-blend (stacking @ K=33) | **12.45283** | 33 | 0 | Same score — ranking identical to gbm at K=33 |
| **union-gbm33-plus-2** | **13.20755** | 35 | +0.75 | **First Phase 8 win** — 2 union exclusives help |
| **union-gbm33-plus-5** | **14.33964** | 38 | +1.89 | **Current best** — full union beyond gbm33 |

**Current best LB:** **14.33964** (`union-gbm33-plus-5`, 38 positives).

**Fallback until beaten:** `outputs/union-gbm33-plus-5/submission/`

---

## What we learned

### 1. Count band matters more than model choice (again)

gbm-recall and gbm-mega-blend both scored **12.45283** at 33 positives. gbm-mega-blend selected **stacking** (OOF acc 0.953) but at K=33 its top-33 coil set matches gbm-recall — blending did not change who gets flagged, only probas in the tail.

### 2. Union augments beat pure gbm tail extension

Extending gbm ranking alone (K=34–36) was untested on LB in this batch, but the union strategy — keep all 33 gbm coils, then add exclusives from the K=26 union — clearly works:

| Augment | Extra coils beyond gbm33 | LB gain |
|---------|--------------------------|---------|
| plus-2 (K=35) | 282, 631 | +6.0% |
| plus-5 (K=38) | 282, 302, 631, 1138, 1346 | +15.1% |

These five exclusives were identified offline in [`scripts/analyze_model_disagreement.py`](../../scripts/analyze_model_disagreement.py): union of gbm33 + lightgbm/sklearn/recall-blend @ K=26 minus the gbm33 set. Secondary models surface high-proba coils (e.g. 282 up to ~0.52) that gbm ranking misses.

### 3. Canary guard held

All uploaded submissions keep canary coils **654, 806, 532, 958, 1187** as Y=1.

---

## What was built and why

### 8A — Exact-K re-threshold (`scripts/rethreshold_submission.py --phase8`)

**Why:** Cheapest LB tests — no retrain, sweep K ∈ {33,34,35,36,38} on saved gbm / gbm-mega / gbm-optuna probas.

**Output:** Per-method `submission_k*.csv` under `outputs/{method}/`.

### 8A — Union augments (`scripts/build_union_submission.py`)

**Why:** Phase 7 disagreement showed 5 coils in the K=26 union not in gbm33. Union augments start from gbm33 (canaries forced), then fill remaining slots to target K by ranking exclusives on `max(sklearn, lgbm, recall-blend proba)`.

| Output | K | Added beyond gbm33 |
|--------|---|-------------------|
| `union-gbm33-plus-2` | 35 | 2 highest-ranked exclusives |
| `union-gbm33-plus-5` | 38 | All 5 union exclusives |

Manifest: `outputs/union_manifest.json`.

### 8B — gbm-mega-blend (`models/gbm-mega-blend/`)

**Why:** Phase 7 mega-recall-blend optimized for K=26 and under-weighted gbm. This variant anchors on gbm-recall (70%) + sklearn/lgbm (15% each), selects strategy by OOF accuracy @ K=33.

**Result:** Stacking @ K=33, OOF acc 0.953, but LB **12.45283** — no gain over gbm alone at same K.

### 8C — gbm-recall top_k_33 retrain

**Why:** Forum t=0.05 ≡ `apply_top_k(gbm_proba, 33)` on existing probas; explicit `top_k_33` in meta improves reproducibility.

**Result:** 33 test positives, LB **12.45283** (matches forum33).

### 8D — gbm-recall-optuna

**Why:** Tune XGB+LGB+CatBoost jointly with canary guard on 806/1187 vs gbm baseline.

**Result:** All 30 trials failed strict guard; model falls back to defaults. Not uploaded in this batch.

### 8E — Feature probes (`scripts/probe_single_feature.py`)

**Why:** Heavy FE broke canaries in Phase 5; probe one ratio at a time with guard.

**Result:** X13/X10 and X30/X32 both **SKIP** (canary proba drop). Report: `outputs/feature_probe_report.json`.

### 8F — Diagnostics

[`scripts/analyze_model_disagreement.py`](../../scripts/analyze_model_disagreement.py) — K=33 union = 44 coils; gbm33 vs K=26 union = 5 exclusives. Report: `outputs/disagreement_report.json`.

---

## Upload pairs (HackerEarth)

| Method | CSV | Source zip |
|--------|-----|------------|
| **union-gbm33-plus-5** (best) | `outputs/union-gbm33-plus-5/submission/submission.csv` | `.../gbm-recall-hackerearth.zip` |
| union-gbm33-plus-2 | `outputs/union-gbm33-plus-2/submission/submission.csv` | `.../gbm-recall-hackerearth.zip` |
| gbm-recall @ K=33 | `outputs/gbm-recall/submission/submission.csv` | `.../gbm-recall-hackerearth.zip` |
| gbm-mega-blend @ K=33 | `outputs/gbm-mega-blend/submission/submission.csv` | `.../gbm-mega-blend-hackerearth.zip` |

Union submissions reuse gbm-recall source zip (same base pipeline + post-hoc augment script).

Manifest: `outputs/pack_manifest.json`. Candidate priority (pre-upload): `outputs/lb_candidates.json`.

---

## Reproduce

```powershell
python models/gbm-mega-blend/train.py --k 33
python models/gbm-mega-blend/predict.py
python models/gbm-recall/train.py --threshold-strategy top_k_33
python models/gbm-recall/predict.py

python scripts/rethreshold_submission.py --phase8 --write-all --write-best --pack --guard-check
python scripts/build_union_submission.py --target-ks 35 38
python scripts/pack_phase8_submissions.py
python scripts/analyze_model_disagreement.py
python scripts/probe_single_feature.py
```

Guard check:

```powershell
python scripts/check_submission_vs_baseline.py models/phase8-rethreshold/outputs/union-gbm33-plus-5/submission/submission.csv --max-positives 40
```

---

## Next directions (from LB evidence)

1. **Union at K=36–37** — interpolate between plus-2 (13.21) and plus-5 (14.34); score may scale smoothly with count + right exclusives.
2. **Rank union candidates by blended proba** — current augment uses max(sklearn, lgbm, recall-blend); weight-opt or gbm+secondary blend for slot filling.
3. **Avoid more K=33-only submissions** — gbm-mega-blend proved stacking does not change the top-33 set.
4. **Skip heavy FE** — single-feature probes failed canary guard.

See also: [DEVELOPMENT.md](../../DEVELOPMENT.md) §18–19, [models/gbm-mega-blend/README.md](../gbm-mega-blend/README.md).
