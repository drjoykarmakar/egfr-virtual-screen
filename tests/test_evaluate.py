import pytest

from src.evaluate import classification_metrics, enrichment_factor, recall_at_fraction


def test_ranking_metrics_reward_perfect_ordering():
    y = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    metrics = classification_metrics(y, scores)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert recall_at_fraction(y, scores, 0.5) == pytest.approx(1.0)
    assert enrichment_factor(y, scores, 0.5) == pytest.approx(2.0)
