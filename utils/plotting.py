"""Shared evaluation plots for Tata Steel hackathon runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ap = average_precision_score(y_true, y_score)
    ax.set_title(f"{title}\nPR-AUC = {ap:.4f}")
    _save(fig, path)


def plot_threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    path: Path,
    *,
    best_threshold: float,
    majority_baseline: float,
) -> None:
    thresholds = np.linspace(0.01, 0.99, 99)
    accs, f1s = [], []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        accs.append(accuracy_score(y_true, pred))
        f1s.append(f1_score(y_true, pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, accs, label="Accuracy", linewidth=2)
    ax.plot(thresholds, f1s, label="F1", linewidth=2, linestyle="--")
    ax.axhline(majority_baseline, color="gray", linestyle=":", label="Majority baseline")
    ax.axvline(best_threshold, color="crimson", linestyle=":", label=f"Best t={best_threshold:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep (OOF)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    path: Path,
    *,
    top_n: int = 15,
    title: str = "Feature importance (top features)",
) -> None:
    order = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(names)), values[::-1], color="steelblue")
    ax.set_yticks(range(len(names)), labels=names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    _save(fig, path)


def plot_fold_scores(fold_scores: dict[str, list[float]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    folds = range(1, len(next(iter(fold_scores.values()))) + 1)
    for metric, scores in fold_scores.items():
        ax.plot(folds, scores, marker="o", label=metric)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(list(folds))
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)
