---
name: tata-steel-model-experiment
description: Scaffolds and runs ML experiments for the Tata Steel hackathon under models/{method}/ with Colab notebooks, stratified CV, and imbalance handling. Use when adding XGBoost, LightGBM, CatBoost, neural nets, ensembles, or new feature pipelines for CoilID/X1–X49 binary classification.
---

# Tata Steel Model Experiment

## When to create a new method folder

Create `models/{kebab-case-name}/` when ANY of these change materially:

- Algorithm family (e.g., XGBoost → CatBoost)
- Feature pipeline (e.g., raw vs engineered vs PCA)
- Ensemble composition
- Validation protocol (e.g., different fold strategy worth comparing)

Do **not** fork a folder for minor hyperparameter tweaks — stay in the same folder and update README.

## Scaffold checklist

```
Method scaffold:
- [ ] models/{name}/README.md (5 required sections)
- [ ] models/{name}/{name}.ipynb (Colab-ready)
- [ ] train.py, predict.py
- [ ] submission/approach.txt (HackerEarth write-up)
- [ ] pack.py (bundle predictions + source for upload)
- [ ] Register results after first CV run
```

### Folder template

```
models/lightgbm-cv/
  README.md
  lightgbm-cv.ipynb
  train.py
  predict.py
  submission/
    approach.txt
    submission.csv          # after pack.py
    {name}-hackerearth.zip
  pack.py                   # bundle for HackerEarth upload
  outputs/
    latest_run.txt
    latest/
    runs/<timestamp>/
      metrics.json
      predictions/submission.csv
      plots/
      artifacts/
```

Shared output helpers: `utils/run_artifacts.py`, `utils/plotting.py`.

## Notebook structure (Colab)

1. **Setup** — `%pip install lightgbm scikit-learn pandas` (method-specific)
2. **GPU check** — if neural net; GBMs are CPU-friendly
3. **Data** — path toggle: local `dataset/` vs Drive mount
4. **Seeds** — `random_state=42` for numpy, sklearn, model
5. **Pipeline** — impute → (optional) feature engineering → model
6. **CV** — `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
7. **Threshold tune** — on OOF probabilities, maximize accuracy
8. **Final fit** — entire train set; predict test
9. **Export** — save run under `outputs/runs/<timestamp>/` (metrics, plots, artifacts, submission)
10. **Latest summary** — copy key files to `outputs/latest/` via `copy_to_latest_summary()`

## Run output standard (local + notebook)

Every training run must persist artifacts inside the method folder:

| Path | Contents |
|------|----------|
| `outputs/runs/<id>/metrics.json` | OOF PR-AUC, accuracy@threshold, fold scores |
| `outputs/runs/<id>/run_config.json` | Data path, hyperparameters, seeds |
| `outputs/runs/<id>/oof_predictions.csv` | CoilID, y_true, oof_proba, oof_pred |
| `outputs/runs/<id>/plots/` | PR curve, threshold sweep, confusion matrix, feature importance |
| `outputs/runs/<id>/artifacts/` | Model + preprocessors + meta |
| `outputs/runs/<id>/predictions/submission.csv` | HackerEarth upload file |
| `outputs/latest/` | Copied snapshot of the most recent run **including `artifacts/`** |

Use a project virtualenv (`.venv/`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
.\.venv\Scripts\Activate.ps1
```

```python
from utils.run_artifacts import create_run_dir, save_metrics, copy_to_latest_summary
from utils.plotting import plot_pr_curve, plot_threshold_sweep

run_dir = create_run_dir(Path("models/xgboost-baseline"))
# ... train ...
save_metrics(run_dir, {"oof_pr_auc": 0.33, "oof_accuracy": 0.956})
copy_to_latest_summary(method_dir, run_dir)
```

Local commands:

```bash
python models/{method}/train.py
python models/{method}/predict.py
```

## Minimal CV pattern

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.impute import SimpleImputer

X = train[features].values
y = train["Y"].astype(int).values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_proba = np.zeros(len(y))
for tr_idx, va_idx in skf.split(X, y):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    imp = SimpleImputer(strategy="median")
    X_tr_i = imp.fit_transform(X_tr)
    X_va_i = imp.transform(X_va)
    # fit model on X_tr_i, y_tr
    # oof_proba[va_idx] = model.predict_proba(X_va_i)[:, 1]

# threshold sweep for accuracy
best_t, best_acc = 0.5, 0.0
for t in np.linspace(0.01, 0.99, 99):
    pred = (oof_proba >= t).astype(int)
    acc = accuracy_score(y, pred)
    if acc > best_acc:
        best_t, best_acc = t, acc

print(f"OOF PR-AUC: {average_precision_score(y, oof_proba):.4f}")
print(f"OOF accuracy @ t={best_t:.3f}: {best_acc:.4f}")
```

## Imbalance defaults by library

| Library | Setting |
|---------|---------|
| LightGBM | `scale_pos_weight = n_neg / n_pos` |
| XGBoost | `scale_pos_weight = n_neg / n_pos` |
| CatBoost | `auto_class_weights='Balanced'` |
| sklearn RF | `class_weight='balanced'` |

## README sections (required)

1. **Method** — algorithm, libs, hyperparameters
2. **Why this approach** — hypothesis for steel tabular data
3. **Technical details** — preprocessing, CV, imbalance, threshold, leakage guards
4. **Results** — OOF PR-AUC, OOF accuracy@threshold, leaderboard if submitted
5. **How to run** — Colab steps with data path

Write with equations and seed values — not one-liners.

## Ensemble pattern

Keep ensemble as its own folder (e.g., `models/gbm-blend/`):

1. Load OOF predictions from base models (or train bases in-notebook)
2. Blend: weighted average of probabilities
3. Tune blend weights + threshold on OOF only
4. Retrain bases on full train; blend test probabilities

## Inference → submission

```python
test_X = imp.transform(test[features].values)
test_proba = model.predict_proba(test_X)[:, 1]
test_pred = (test_proba >= best_t).astype(int)
sub = pd.DataFrame({"CoilID": test["CoilID"], "Y": test_pred})
run_dir = Path("models/{method}/outputs/runs/<timestamp>")
sub.to_csv(run_dir / "predictions/submission.csv", index=False)
```

Then validate and package: [tata-steel-submission](../tata-steel-submission/SKILL.md).

```bash
python .cursor/skills/tata-steel-submission/scripts/pack_submission.py models/{name}
```

## Suggested experiment order

1. `xgboost-baseline` — strong default, fast
2. `lightgbm-cv` — tuned GBM + threshold sweep
3. `catboost-cv` — handles categoricals if you bin features
4. `gbm-blend` — ensemble top 2–3
5. Optional `neural-net-mlp` — only if GBMs plateau
