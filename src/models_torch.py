"""Small PyTorch fingerprint MLP for EGFR activity classification.

The model intentionally shares the exact Morgan+descriptor representation used
by the Random Forest baseline. Morgan bits remain binary; only the continuous
RDKit descriptor suffix is standardized, and the standardizer is fit on the
training split only.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import classification_metrics


@dataclass(frozen=True)
class DescriptorStandardizer:
    """Training-fit standardization parameters for descriptor columns only."""

    n_fingerprint_features: int
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X_train: np.ndarray, n_fingerprint_features: int) -> "DescriptorStandardizer":
        X = _validate_feature_matrix(X_train)
        n_fp = int(n_fingerprint_features)
        if not 0 <= n_fp <= X.shape[1]:
            raise ValueError(
                f"n_fingerprint_features must be in [0, {X.shape[1]}]; got {n_fp}."
            )
        descriptor_values = X[:, n_fp:]
        if descriptor_values.shape[1] == 0:
            return cls(
                n_fingerprint_features=n_fp,
                mean=np.empty(0, dtype=np.float32),
                scale=np.empty(0, dtype=np.float32),
            )
        mean = descriptor_values.mean(axis=0, dtype=np.float64)
        scale = descriptor_values.std(axis=0, dtype=np.float64)
        # Constant descriptors carry no scale information. Leaving their scale at
        # one maps them to zero after centering without introducing infinities.
        scale = np.where(scale > 0.0, scale, 1.0)
        return cls(
            n_fingerprint_features=n_fp,
            mean=np.asarray(mean, dtype=np.float32),
            scale=np.asarray(scale, dtype=np.float32),
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = _validate_feature_matrix(X).astype(np.float32, copy=True)
        expected_descriptors = values.shape[1] - self.n_fingerprint_features
        if expected_descriptors != len(self.mean):
            raise ValueError(
                "Feature dimensionality does not match fitted descriptor standardizer: "
                f"expected {self.n_fingerprint_features + len(self.mean)}, got {values.shape[1]}."
            )
        if expected_descriptors:
            values[:, self.n_fingerprint_features :] = (
                values[:, self.n_fingerprint_features :] - self.mean
            ) / self.scale
        return values

    def state_dict(self) -> dict[str, object]:
        return {
            "n_fingerprint_features": self.n_fingerprint_features,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "DescriptorStandardizer":
        return cls(
            n_fingerprint_features=int(state["n_fingerprint_features"]),
            mean=np.asarray(state["mean"], dtype=np.float32),
            scale=np.asarray(state["scale"], dtype=np.float32),
        )


class FingerprintMLP(nn.Module):
    """Compact feed-forward binary classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (512, 128),
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("All hidden dimensions must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_dims:
            width = int(width)
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.dropout = float(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass
class TorchFit:
    model: FingerprintMLP
    standardizer: DescriptorStandardizer
    best_epoch: int
    validation_metrics: dict[str, float]
    history: list[dict[str, float | int]]
    pos_weight: float


def _validate_feature_matrix(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix; got shape {values.shape}.")
    if values.size and not np.isfinite(values).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")
    return values


def _validate_binary_labels(labels: Iterable[int], *, require_both: bool) -> np.ndarray:
    y = np.asarray(list(labels), dtype=np.int64)
    if y.ndim != 1:
        raise ValueError("Labels must be one-dimensional.")
    unique = set(np.unique(y).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1; got {sorted(unique)}.")
    if require_both and unique != {0, 1}:
        raise ValueError("Training split must contain both activity classes.")
    return y


def set_torch_seed(seed: int) -> None:
    """Set deterministic seeds used by this small CPU training loop."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CPU operations used here are deterministic. These flags make accidental
    # future device changes fail loudly instead of silently changing semantics.
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass


def training_pos_weight(y_train: Iterable[int]) -> float:
    """Return n_negative / n_positive using training labels only."""

    y = _validate_binary_labels(y_train, require_both=True)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return float(negatives / positives)


def predict_active_probability(
    model: FingerprintMLP,
    standardizer: DescriptorStandardizer,
    X: np.ndarray,
    *,
    batch_size: int = 1024,
    device: str = "cpu",
) -> np.ndarray:
    """Predict active probabilities without modifying training state."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    transformed = standardizer.transform(X)
    tensor = torch.from_numpy(transformed)
    loader = DataLoader(TensorDataset(tensor), batch_size=int(batch_size), shuffle=False)
    model = model.to(device)
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for (features,) in loader:
            logits = model(features.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    if not probabilities:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(probabilities).astype(np.float32, copy=False)


def fit_fingerprint_mlp(
    X_train: np.ndarray,
    y_train: Iterable[int],
    X_validation: np.ndarray,
    y_validation: Iterable[int],
    *,
    n_fingerprint_features: int,
    hidden_dims: Sequence[int] = (512, 128),
    dropout: float = 0.20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    early_stopping_patience: int = 12,
    selection_metric: str = "auprc",
    device: str = "cpu",
    seed: int = 2026,
) -> TorchFit:
    """Train with validation-only early stopping and return the best epoch.

    Continuous descriptor columns are standardized from ``X_train`` only. The
    validation split is used only for early stopping/model selection; no test
    data are accepted by this function.
    """

    X_train = _validate_feature_matrix(X_train)
    X_validation = _validate_feature_matrix(X_validation)
    if X_train.shape[1] != X_validation.shape[1]:
        raise ValueError("Training and validation feature dimensions differ.")
    y_train_array = _validate_binary_labels(y_train, require_both=True)
    y_validation_array = _validate_binary_labels(y_validation, require_both=False)
    if len(X_train) != len(y_train_array) or len(X_validation) != len(y_validation_array):
        raise ValueError("Feature and label row counts must match.")
    if len(X_validation) == 0:
        raise ValueError("Validation split must not be empty.")
    if max_epochs <= 0 or early_stopping_patience <= 0 or batch_size <= 0:
        raise ValueError("max_epochs, early_stopping_patience, and batch_size must be positive.")

    set_torch_seed(seed)
    standardizer = DescriptorStandardizer.fit(X_train, n_fingerprint_features)
    X_train_scaled = standardizer.transform(X_train)
    X_validation_scaled = standardizer.transform(X_validation)

    train_features = torch.from_numpy(X_train_scaled)
    train_targets = torch.from_numpy(y_train_array.astype(np.float32))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )

    model = FingerprintMLP(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    positive_weight = training_pos_weight(y_train_array)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )

    validation_tensor = torch.from_numpy(X_validation_scaled).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    best_value = -np.inf
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        running_loss = 0.0
        rows_seen = 0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            batch_rows = int(len(targets))
            running_loss += float(loss.detach().cpu()) * batch_rows
            rows_seen += batch_rows

        model.eval()
        with torch.no_grad():
            validation_scores = torch.sigmoid(model(validation_tensor)).cpu().numpy()
        metrics = classification_metrics(y_validation_array, validation_scores)
        metric_value = float(metrics.get(selection_metric, float("nan")))
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(rows_seen, 1),
                **metrics,
            }
        )

        improved = np.isfinite(metric_value) and metric_value > best_value + 1e-12
        if improved:
            best_value = metric_value
            best_epoch = epoch
            best_metrics = metrics
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(early_stopping_patience):
            break

    if best_state is None or best_metrics is None:
        raise ValueError(
            f"Selection metric {selection_metric!r} was undefined for every validation epoch."
        )
    model.load_state_dict(best_state)
    model = model.to("cpu")
    return TorchFit(
        model=model,
        standardizer=standardizer,
        best_epoch=best_epoch,
        validation_metrics=best_metrics,
        history=history,
        pos_weight=positive_weight,
    )


def save_torch_checkpoint(
    path: str | Path,
    fit: TorchFit,
    *,
    feature_config: dict[str, object],
) -> None:
    """Save model and preprocessing state needed for later library scoring."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": fit.model.state_dict(),
        "input_dim": fit.model.input_dim,
        "hidden_dims": list(fit.model.hidden_dims),
        "dropout": fit.model.dropout,
        "standardizer": fit.standardizer.state_dict(),
        "best_epoch": fit.best_epoch,
        "validation_metrics": fit.validation_metrics,
        "pos_weight": fit.pos_weight,
        "feature_config": dict(feature_config),
    }
    torch.save(payload, output)


def load_torch_checkpoint(
    path: str | Path,
    *,
    map_location: str = "cpu",
) -> tuple[FingerprintMLP, DescriptorStandardizer, dict[str, object]]:
    """Load a saved MLP checkpoint for evaluation or screening."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    model = FingerprintMLP(
        input_dim=int(payload["input_dim"]),
        hidden_dims=payload["hidden_dims"],
        dropout=float(payload["dropout"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    standardizer = DescriptorStandardizer.from_state_dict(payload["standardizer"])
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state_dict", "standardizer"}
    }
    return model, standardizer, metadata
