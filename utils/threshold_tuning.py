"""Threshold tuning strategies for imbalanced binary classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    accuracy: float
    recall: float
    precision: float
    n_predicted_positive: int
    false_positives: int
    false_negatives: int
    strategy: str

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "recall": self.recall,
            "precision": self.precision,
            "n_predicted_positive": self.n_predicted_positive,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "strategy": self.strategy,
        }


def _metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray, t: float) -> ThresholdResult:
    pred = (proba >= t).astype(int)
    y = np.asarray(y_true, dtype=int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    recall = recall_score(y, pred, zero_division=0)
    precision = precision_score(y, pred, zero_division=0)
    return ThresholdResult(
        threshold=float(t),
        accuracy=float(accuracy_score(y, pred)),
        recall=float(recall),
        precision=float(precision),
        n_predicted_positive=int(pred.sum()),
        false_positives=fp,
        false_negatives=fn,
        strategy="",
    )


def sweep_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    n_steps: int = 99,
    t_min: float = 0.01,
    t_max: float = 0.99,
) -> list[ThresholdResult]:
    results: list[ThresholdResult] = []
    for t in np.linspace(t_min, t_max, n_steps):
        m = _metrics_at_threshold(y_true, proba, float(t))
        results.append(m)
    return results


def tune_max_accuracy(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    n_steps: int = 99,
) -> ThresholdResult:
    best: ThresholdResult | None = None
    for m in sweep_thresholds(y_true, proba, n_steps=n_steps):
        if best is None or m.accuracy > best.accuracy:
            best = m
        elif abs(m.accuracy - best.accuracy) < 1e-9 and m.n_predicted_positive > best.n_predicted_positive:
            best = m
    assert best is not None
    return ThresholdResult(**{**best.__dict__, "strategy": "max_accuracy"})


def tune_recall_first(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    min_recall: float = 1.0,
    n_steps: int = 199,
    t_min: float = 0.001,
    t_max: float = 0.99,
) -> ThresholdResult:
    """Smallest threshold achieving min_recall; tie-break by accuracy then fewer FP."""
    y = np.asarray(y_true, dtype=int)
    candidates: list[ThresholdResult] = []
    for t in np.linspace(t_min, t_max, n_steps):
        m = _metrics_at_threshold(y, proba, float(t))
        if m.recall >= min_recall - 1e-9:
            candidates.append(m)

    if not candidates:
        m = _metrics_at_threshold(y, proba, t_min)
        return ThresholdResult(**{**m.__dict__, "strategy": "recall_first_fallback"})

    # Highest threshold that still meets recall — fewest false positives while keeping recall
    best = max(candidates, key=lambda m: (m.threshold, m.accuracy, -m.false_positives))
    return ThresholdResult(**{**best.__dict__, "strategy": "recall_first"})


def tune_target_fpr(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    max_fpr: float = 0.03,
    n_steps: int = 199,
    t_min: float = 0.001,
    t_max: float = 0.99,
) -> ThresholdResult:
    """Among thresholds with FPR <= max_fpr, maximize recall then accuracy."""
    y = np.asarray(y_true, dtype=int)
    n_neg = max(int((y == 0).sum()), 1)
    candidates: list[ThresholdResult] = []

    for t in np.linspace(t_min, t_max, n_steps):
        m = _metrics_at_threshold(y, proba, float(t))
        fpr = m.false_positives / n_neg
        if fpr <= max_fpr + 1e-9:
            candidates.append(m)

    if not candidates:
        m = tune_max_accuracy(y, proba, n_steps=n_steps)
        return ThresholdResult(**{**m.__dict__, "strategy": "target_fpr_fallback_max_accuracy"})

    candidates.sort(key=lambda m: (-m.recall, -m.accuracy, m.false_positives))
    best = candidates[0]
    return ThresholdResult(**{**best.__dict__, "strategy": "target_fpr"})


def select_recall_oriented_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    min_recall: float = 1.0,
    max_fpr: float = 0.03,
    forum_fixed_t: float = 0.05,
    max_positive_rate: float = 0.12,
) -> ThresholdResult:
    """Recall-first with fallbacks: 100% OOF recall, else FPR cap, else forum t=0.05."""
    y = np.asarray(y_true, dtype=int)
    max_pos = int(max_positive_rate * len(y))

    recall = tune_recall_first(y, proba, min_recall=min_recall)
    if (
        recall.recall >= min_recall - 1e-9
        and recall.false_negatives == 0
        and recall.n_predicted_positive <= max_pos
    ):
        return recall

    fpr = tune_target_fpr(y, proba, max_fpr=max_fpr)
    fixed = _metrics_at_threshold(y, proba, forum_fixed_t)
    fixed = ThresholdResult(**{**fixed.__dict__, "strategy": "fixed_t_0.05"})

    if fixed.recall >= fpr.recall + 0.05 and fixed.accuracy >= fpr.accuracy - 0.015:
        return fixed

    candidates = [fpr, fixed]
    if recall.recall >= fpr.recall and recall.n_predicted_positive <= max_pos:
        candidates.append(recall)

    candidates.sort(key=lambda m: (m.accuracy, m.recall, -m.false_positives), reverse=True)
    return candidates[0]


def apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba >= threshold).astype(int)


def format_result(m: ThresholdResult) -> str:
    return (
        f"[{m.strategy}] t={m.threshold:.4f} acc={m.accuracy:.4f} "
        f"rec={m.recall:.4f} prec={m.precision:.4f} "
        f"pos={m.n_predicted_positive} fp={m.false_positives} fn={m.false_negatives}"
    )
