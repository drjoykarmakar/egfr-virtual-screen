import numpy as np
import pytest
import torch

from src.models_torch import (
    DescriptorStandardizer,
    FingerprintMLP,
    fit_fingerprint_mlp,
    load_torch_checkpoint,
    predict_active_probability,
    save_torch_checkpoint,
)


def _toy_data():
    # Four binary fingerprint bits + two continuous descriptors. The label is
    # deliberately simple so a tiny deterministic network can learn it quickly.
    rng = np.random.default_rng(7)
    fingerprints = rng.integers(0, 2, size=(80, 4)).astype(np.float32)
    descriptors = rng.normal(size=(80, 2)).astype(np.float32)
    y = ((fingerprints[:, 0] + fingerprints[:, 1] + (descriptors[:, 0] > 0)) >= 2).astype(int)
    X = np.hstack([fingerprints, descriptors]).astype(np.float32)
    return X, y


def test_descriptor_standardizer_uses_training_statistics_only():
    train = np.array(
        [
            [0, 1, 10.0, 100.0],
            [1, 0, 20.0, 200.0],
        ],
        dtype=np.float32,
    )
    validation = np.array([[1, 1, 1000.0, 2000.0]], dtype=np.float32)
    standardizer = DescriptorStandardizer.fit(train, n_fingerprint_features=2)

    transformed_train = standardizer.transform(train)
    transformed_validation = standardizer.transform(validation)

    assert np.allclose(transformed_train[:, :2], train[:, :2])
    assert np.allclose(transformed_train[:, 2:].mean(axis=0), 0.0)
    # If validation had leaked into the fitted statistics this value would be small.
    assert transformed_validation[0, 2] > 100.0


def test_mlp_training_is_deterministic_and_probabilities_are_valid():
    X, y = _toy_data()
    kwargs = dict(
        n_fingerprint_features=4,
        hidden_dims=[16, 8],
        dropout=0.0,
        batch_size=16,
        learning_rate=0.01,
        weight_decay=0.0,
        max_epochs=20,
        early_stopping_patience=5,
        seed=11,
    )
    first = fit_fingerprint_mlp(X[:60], y[:60], X[60:], y[60:], **kwargs)
    second = fit_fingerprint_mlp(X[:60], y[:60], X[60:], y[60:], **kwargs)

    first_scores = predict_active_probability(first.model, first.standardizer, X[60:])
    second_scores = predict_active_probability(second.model, second.standardizer, X[60:])
    assert np.all((0.0 <= first_scores) & (first_scores <= 1.0))
    assert np.allclose(first_scores, second_scores, atol=1e-7)
    assert first.best_epoch == second.best_epoch


def test_checkpoint_roundtrip_preserves_scores(tmp_path):
    X, y = _toy_data()
    fit = fit_fingerprint_mlp(
        X[:60],
        y[:60],
        X[60:],
        y[60:],
        n_fingerprint_features=4,
        hidden_dims=[8],
        dropout=0.0,
        batch_size=20,
        learning_rate=0.01,
        max_epochs=8,
        early_stopping_patience=3,
        seed=5,
    )
    path = tmp_path / "model.pt"
    save_torch_checkpoint(path, fit, feature_config={"radius": 2, "n_bits": 4})
    model, standardizer, metadata = load_torch_checkpoint(path)

    expected = predict_active_probability(fit.model, fit.standardizer, X[60:])
    observed = predict_active_probability(model, standardizer, X[60:])
    assert np.allclose(expected, observed, atol=1e-7)
    assert metadata["best_epoch"] == fit.best_epoch


def test_mlp_forward_shape():
    model = FingerprintMLP(input_dim=10, hidden_dims=[4], dropout=0.1)
    logits = model(torch.zeros((3, 10), dtype=torch.float32))
    assert logits.shape == (3,)
