"""Evaluation utilities for binary EGFR activity ranking."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _top_n(n_items: int, fraction: float) -> int:
    if not 0 < fraction <= 1:
        raise ValueError(f"Fraction must be in (0, 1]; got {fraction}.")
    return min(n_items, max(1, int(math.ceil(n_items * fraction))))


def recall_at_fraction(y_true: Iterable[int], y_score: Iterable[float], fraction: float) -> float:
    """Fraction of all positives recovered in the top-scoring library fraction."""

    y = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(y_score), dtype=float)
    positives = int(y.sum())
    if len(y) == 0 or positives == 0:
        return float("nan")
    n_top = _top_n(len(y), fraction)
    order = np.argsort(-scores, kind="stable")[:n_top]
    return float(y[order].sum() / positives)


def enrichment_factor(y_true: Iterable[int], y_score: Iterable[float], fraction: float) -> float:
    """Enrichment factor in the top-scoring fraction relative to random ranking."""

    y = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(y_score), dtype=float)
    if len(y) == 0:
        return float("nan")
    prevalence = float(y.mean())
    if prevalence <= 0:
        return float("nan")
    n_top = _top_n(len(y), fraction)
    order = np.argsort(-scores, kind="stable")[:n_top]
    top_precision = float(y[order].mean())
    return top_precision / prevalence


def classification_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
) -> dict[str, float]:
    """Compute ranking-focused classification metrics.

    AUROC is undefined if only one class is present; AUPRC is undefined here
    when there are no positives. Undefined metrics are returned as NaN rather
    than replaced with misleading constants.
    """

    y = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(y_score), dtype=float)
    if y.shape != scores.shape:
        raise ValueError("y_true and y_score must have identical shapes.")
    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinite values.")

    unique = np.unique(y)
    auroc = float(roc_auc_score(y, scores)) if len(unique) == 2 else float("nan")
    auprc = float(average_precision_score(y, scores)) if int(y.sum()) > 0 else float("nan")
    return {
        "auroc": auroc,
        "auprc": auprc,
        "recall_at_1pct": recall_at_fraction(y, scores, 0.01),
        "recall_at_5pct": recall_at_fraction(y, scores, 0.05),
        "ef_at_1pct": enrichment_factor(y, scores, 0.01),
        "ef_at_5pct": enrichment_factor(y, scores, 0.05),
    }
