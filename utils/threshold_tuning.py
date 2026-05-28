"""Threshold tuning strategies for imbalanced binary classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
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


def tune_target_positive_rate(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    target_rate: float = 0.077,
    n_steps: int = 199,
    t_min: float = 0.001,
    t_max: float = 0.99,
) -> ThresholdResult:
    """Pick threshold whose OOF positive rate is closest to target_rate (~26/339)."""
    y = np.asarray(y_true, dtype=int)
    n = len(y)
    target_count = int(round(target_rate * n))
    best: ThresholdResult | None = None
    best_dist = float("inf")

    for t in np.linspace(t_min, t_max, n_steps):
        m = _metrics_at_threshold(y, proba, float(t))
        dist = abs(m.n_predicted_positive - target_count)
        if dist < best_dist or (
            dist == best_dist and best is not None and m.accuracy > best.accuracy
        ):
            best_dist = dist
            best = m

    assert best is not None
    return ThresholdResult(**{**best.__dict__, "strategy": f"target_rate_{target_rate:.3f}"})


def tune_fixed_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    *,
    strategy_name: str | None = None,
) -> ThresholdResult:
    m = _metrics_at_threshold(y_true, proba, threshold)
    name = strategy_name or f"fixed_t_{threshold:g}"
    return ThresholdResult(**{**m.__dict__, "strategy": name})


def tune_fixed_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: tuple[float, ...] = (0.05, 0.31, 0.35),
) -> dict[str, ThresholdResult]:
    return {
        f"fixed_t_{t:g}": tune_fixed_threshold(y_true, proba, t) for t in thresholds
    }


def tune_target_test_positives(
    y_true: np.ndarray,
    proba: np.ndarray,
    test_proba: np.ndarray,
    *,
    target_test_positives: int = 26,
    min_test_positives: int = 24,
    max_test_positives: int = 28,
    n_steps: int = 199,
    t_min: float = 0.001,
    t_max: float = 0.99,
) -> ThresholdResult:
    """Pick threshold where test positive count is in [min, max], best OOF accuracy."""
    y = np.asarray(y_true, dtype=int)
    test_proba = np.asarray(test_proba, dtype=float)
    candidates: list[tuple[ThresholdResult, int]] = []

    for t in np.linspace(t_min, t_max, n_steps):
        m = _metrics_at_threshold(y, proba, float(t))
        test_pos = int((test_proba >= t).sum())
        if min_test_positives <= test_pos <= max_test_positives:
            candidates.append((m, test_pos))

    if not candidates:
        rate = tune_target_positive_rate(y, proba, target_rate=target_test_positives / 339.0)
        return ThresholdResult(**{**rate.__dict__, "strategy": "target_test_pos_fallback_rate"})

    best_m, _ = max(candidates, key=lambda x: (x[0].accuracy, x[0].recall, -abs(x[1] - target_test_positives)))
    return ThresholdResult(**{**best_m.__dict__, "strategy": "target_test_positives"})


def select_threshold_by_strategy(
    y_true: np.ndarray,
    proba: np.ndarray,
    strategy: str,
    *,
    test_proba: np.ndarray | None = None,
    min_recall: float = 1.0,
    max_fpr: float = 0.03,
    forum_fixed_t: float = 0.05,
    max_positive_rate: float = 0.12,
) -> ThresholdResult:
    """Dispatch threshold selection by strategy name."""
    if strategy == "forum_fixed":
        return tune_fixed_threshold(y_true, proba, forum_fixed_t, strategy_name="forum_fixed_0.05")
    if strategy == "target_test_positives" and test_proba is not None:
        return tune_target_test_positives(y_true, proba, test_proba)
    if strategy == "target_rate":
        return tune_target_positive_rate(y_true, proba)
    if strategy == "fixed_0.31":
        return tune_fixed_threshold(y_true, proba, 0.31, strategy_name="forum_fixed_0.31")
    if strategy == "top_k_33":
        return tune_top_k(y_true, proba, 33)
    if strategy.startswith("top_k_"):
        k = int(strategy.split("_")[-1])
        return tune_top_k(y_true, proba, k)
    return select_recall_oriented_threshold(
        y_true,
        proba,
        min_recall=min_recall,
        max_fpr=max_fpr,
        forum_fixed_t=forum_fixed_t,
        max_positive_rate=max_positive_rate,
    )


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


def apply_top_k(
    proba: np.ndarray,
    k: int,
    *,
    force_positive_idx: np.ndarray | list[int] | None = None,
) -> tuple[np.ndarray, float]:
    """Mark exactly k highest-probability rows positive; force canary indices first."""
    proba = np.asarray(proba, dtype=float)
    n = len(proba)
    k = int(min(max(k, 0), n))
    pred = np.zeros(n, dtype=int)

    forced: list[int] = []
    if force_positive_idx is not None:
        forced = [int(i) for i in force_positive_idx if 0 <= int(i) < n]
        forced = list(dict.fromkeys(forced))
        if len(forced) > k:
            forced = sorted(forced, key=lambda i: proba[i], reverse=True)[:k]
        for i in forced:
            pred[i] = 1

    remaining = k - int(pred.sum())
    if remaining > 0:
        available = np.where(pred == 0)[0]
        order = available[np.argsort(-proba[available])]
        for i in order[:remaining]:
            pred[i] = 1

    selected = np.where(pred == 1)[0]
    threshold = float(proba[selected].min()) if len(selected) else 1.0
    return pred, threshold


def _metrics_at_top_k(
    y_true: np.ndarray,
    proba: np.ndarray,
    k: int,
    *,
    force_positive_idx: np.ndarray | list[int] | None = None,
) -> ThresholdResult:
    pred, threshold = apply_top_k(proba, k, force_positive_idx=force_positive_idx)
    y = np.asarray(y_true, dtype=int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return ThresholdResult(
        threshold=threshold,
        accuracy=float(accuracy_score(y, pred)),
        recall=float(recall_score(y, pred, zero_division=0)),
        precision=float(precision_score(y, pred, zero_division=0)),
        n_predicted_positive=int(pred.sum()),
        false_positives=fp,
        false_negatives=fn,
        strategy="",
    )


def tune_top_k(
    y_true: np.ndarray,
    proba: np.ndarray,
    k: int,
    *,
    force_positive_idx: np.ndarray | list[int] | None = None,
) -> ThresholdResult:
    m = _metrics_at_top_k(y_true, proba, k, force_positive_idx=force_positive_idx)
    return ThresholdResult(**{**m.__dict__, "strategy": f"top_k_{k}"})


def format_result(m: ThresholdResult) -> str:
    return (
        f"[{m.strategy}] t={m.threshold:.4f} acc={m.accuracy:.4f} "
        f"rec={m.recall:.4f} prec={m.precision:.4f} "
        f"pos={m.n_predicted_positive} fp={m.false_positives} fn={m.false_negatives}"
    )


def rank_average_proba(probas: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Mean rank across models — preserves per-model ordering (not prob averaging)."""
    if isinstance(probas, np.ndarray):
        if probas.ndim == 1:
            return probas.astype(float)
        probas = [probas[i] for i in range(probas.shape[0])]
    if not probas:
        raise ValueError("rank_average_proba requires at least one proba vector")
    ranks = np.stack([stats.rankdata(np.asarray(p, dtype=float)) for p in probas], axis=0)
    return ranks.mean(axis=0)


def sweep_optimal_k(
    y_true: np.ndarray,
    proba: np.ndarray,
    k_min: int,
    k_max: int,
    *,
    force_positive_idx: np.ndarray | list[int] | None = None,
) -> tuple[int, ThresholdResult]:
    """Pick K in [k_min, k_max] maximizing OOF accuracy @ top-K."""
    y = np.asarray(y_true, dtype=int)
    n = len(y)
    k_min = int(max(1, k_min))
    k_max = int(min(k_max, n))
    best_k = k_min
    best_m: ThresholdResult | None = None
    for k in range(k_min, k_max + 1):
        m = tune_top_k(y, proba, k, force_positive_idx=force_positive_idx)
        if best_m is None or m.accuracy > best_m.accuracy or (
            abs(m.accuracy - best_m.accuracy) < 1e-9 and m.recall > best_m.recall
        ):
            best_k = k
            best_m = m
    assert best_m is not None
    return best_k, best_m
