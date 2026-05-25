"""Train XGBoost baseline with stratified CV; save metrics, plots, and artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, classification_report
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plotting import (  # noqa: E402
    plot_confusion_matrix,
    plot_feature_importance,
    plot_fold_scores,
    plot_pr_curve,
    plot_threshold_sweep,
)
from utils.run_artifacts import (  # noqa: E402
    copy_to_latest_summary,
    create_run_dir,
    print_saved_artifacts,
    save_metrics,
    save_run_config,
    write_artifacts_manifest,
)

METHOD_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_SPLITS = 5
FEATURES = [f"X{i}" for i in range(1, 50)]

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (proba >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_t, best_acc = float(t), float(acc)
    return best_t, best_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost baseline and save run outputs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id")
    parser.add_argument("--run-dir", type=Path, default=None, help="Override run directory")
    args = parser.parse_args()

    run_dir = args.run_dir or create_run_dir(METHOD_DIR, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    plots_dir = run_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    X = train[FEATURES].values
    y = train["Y"].astype(int).values
    coil_ids = train["CoilID"].values

    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    majority_baseline = float((y == 0).mean())
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y))
    fold_pr_aucs: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[tr_idx])
        X_va = imp.transform(X[va_idx])
        model = XGBClassifier(**params)
        model.fit(X_tr, y[tr_idx], verbose=False)
        oof_proba[va_idx] = model.predict_proba(X_va)[:, 1]
        fold_pr = average_precision_score(y[va_idx], oof_proba[va_idx])
        fold_pr_aucs.append(float(fold_pr))
        print(f"fold {fold} PR-AUC: {fold_pr:.4f}")

    best_t, best_acc = tune_threshold(y, oof_proba)
    oof_pr_auc = float(average_precision_score(y, oof_proba))
    oof_pred = (oof_proba >= best_t).astype(int)
    report = classification_report(y, oof_pred, output_dict=True, zero_division=0)

    pd.DataFrame(
        {"CoilID": coil_ids, "y_true": y, "oof_proba": oof_proba, "oof_pred": oof_pred}
    ).to_csv(run_dir / "oof_predictions.csv", index=False)

    imp_full = SimpleImputer(strategy="median")
    X_full = imp_full.fit_transform(X)
    final_model = XGBClassifier(**params)
    final_model.fit(X_full, y, verbose=False)

    model_json = artifacts_dir / "xgb_model.json"
    model_joblib = artifacts_dir / "model.joblib"
    imputer_path = artifacts_dir / "imputer.joblib"
    meta_path = artifacts_dir / "meta.joblib"

    final_model.save_model(str(model_json))
    joblib.dump(final_model, model_joblib)
    joblib.dump(imp_full, imputer_path)

    meta = {
        "method": "xgboost-baseline",
        "threshold": best_t,
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "n_splits": N_SPLITS,
        "model_files": {
            "xgb_native": model_json.name,
            "sklearn_wrapper": model_joblib.name,
            "imputer": imputer_path.name,
            "meta": meta_path.name,
        },
    }
    joblib.dump(meta, meta_path)
    write_artifacts_manifest(artifacts_dir)

    metrics = {
        "method": "xgboost-baseline",
        "oof_pr_auc": oof_pr_auc,
        "oof_accuracy": best_acc,
        "best_threshold": best_t,
        "majority_baseline_accuracy": majority_baseline,
        "fold_pr_auc": fold_pr_aucs,
        "classification_report": report,
        "train_rows": len(y),
        "positive_rate": float(y.mean()),
    }
    save_metrics(run_dir, metrics)
    save_run_config(
        run_dir,
        {
            "data_dir": str(args.data_dir.resolve()),
            "run_dir": str(run_dir.resolve()),
            "hyperparameters": params,
        },
    )

    plot_pr_curve(y, oof_proba, plots_dir / "oof_pr_curve.png", "OOF precision-recall")
    plot_threshold_sweep(
        y,
        oof_proba,
        plots_dir / "threshold_sweep.png",
        best_threshold=best_t,
        majority_baseline=majority_baseline,
    )
    plot_confusion_matrix(
        y,
        oof_pred,
        plots_dir / "confusion_matrix_oof.png",
        f"OOF confusion matrix (t={best_t:.3f})",
    )
    plot_feature_importance(
        FEATURES,
        final_model.feature_importances_,
        plots_dir / "feature_importance.png",
    )
    plot_fold_scores({"PR-AUC": fold_pr_aucs}, plots_dir / "fold_pr_auc.png", "CV fold PR-AUC")

    copy_to_latest_summary(METHOD_DIR, run_dir)

    print(f"OOF PR-AUC: {oof_pr_auc:.4f}")
    print(f"OOF accuracy @ t={best_t:.3f}: {best_acc:.4f}")
    print(f"Run saved to: {run_dir}")
    print(f"Model artifacts: {artifacts_dir}")
    print_saved_artifacts(artifacts_dir)
    print(f"Latest (incl. models): {METHOD_DIR / 'outputs' / 'latest' / 'artifacts'}")


if __name__ == "__main__":
    main()
