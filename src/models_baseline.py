"""Random Forest baseline with validation-only light tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.evaluate import classification_metrics


@dataclass(frozen=True)
class BaselineFit:
    model: RandomForestClassifier
    best_params: dict[str, Any]
    validation_metrics: dict[str, float]
    candidate_results: list[dict[str, Any]]


def build_random_forest(
    params: dict[str, Any],
    *,
    random_state: int = 2026,
    class_weight: str | dict | None = "balanced",
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """Construct the transparent baseline classifier."""

    merged: dict[str, Any] = {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "class_weight": class_weight,
        "n_jobs": n_jobs,
        "random_state": int(random_state),
    }
    merged.update(params)
    # Keep experiment-control parameters authoritative even if a tuning dict
    # accidentally contains them.
    merged["random_state"] = int(random_state)
    merged["class_weight"] = class_weight
    merged["n_jobs"] = int(n_jobs)
    return RandomForestClassifier(**merged)


def fit_baseline_with_validation(
    X_train: np.ndarray,
    y_train: Iterable[int],
    X_validation: np.ndarray,
    y_validation: Iterable[int],
    candidates: Sequence[dict[str, Any]],
    *,
    random_state: int = 2026,
    class_weight: str | dict | None = "balanced",
    n_jobs: int = -1,
    selection_metric: str = "auprc",
) -> BaselineFit:
    """Fit candidate forests and select using validation data only."""

    if not candidates:
        raise ValueError("At least one baseline candidate configuration is required.")
    y_train_array = np.asarray(list(y_train), dtype=int)
    y_validation_array = np.asarray(list(y_validation), dtype=int)
    if len(np.unique(y_train_array)) < 2:
        raise ValueError("Training split must contain both activity classes.")

    fitted: list[tuple[RandomForestClassifier, dict[str, Any], dict[str, float]]] = []
    results: list[dict[str, Any]] = []
    for candidate_index, params in enumerate(candidates):
        model = build_random_forest(
            dict(params),
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )
        model.fit(X_train, y_train_array)
        validation_scores = model.predict_proba(X_validation)[:, 1]
        metrics = classification_metrics(y_validation_array, validation_scores)
        record = {"candidate": candidate_index, **params, **metrics}
        results.append(record)
        fitted.append((model, dict(params), metrics))

    def metric_key(item: tuple[RandomForestClassifier, dict[str, Any], dict[str, float]]) -> float:
        value = float(item[2].get(selection_metric, float("nan")))
        return value if np.isfinite(value) else -np.inf

    best_model, best_params, best_metrics = max(fitted, key=metric_key)
    if not np.isfinite(metric_key((best_model, best_params, best_metrics))):
        raise ValueError(
            f"Selection metric {selection_metric!r} is undefined for every candidate."
        )
    return BaselineFit(
        model=best_model,
        best_params=best_params,
        validation_metrics=best_metrics,
        candidate_results=results,
    )
