import pandas as pd

from scripts.make_report import build_model_comparison, select_primary_scaffold_model


def _row(protocol, model, value, validation=None):
    return {
        "split_protocol": protocol,
        "model": model,
        "auroc": value,
        "auprc": value,
        "recall_at_1pct": value,
        "recall_at_5pct": value,
        "ef_at_1pct": value,
        "ef_at_5pct": value,
        "validation_selection_metric": value if validation is None else validation,
        "selection_metric": "auprc",
    }


def test_model_comparison_has_expected_order():
    baseline = pd.DataFrame(
        [
            _row("scaffold", "RandomForestClassifier", 0.6),
            _row("random", "RandomForestClassifier", 0.8),
        ]
    )
    torch_metrics = pd.DataFrame(
        [
            _row("scaffold", "TorchFingerprintMLP", 0.62),
            _row("random", "TorchFingerprintMLP", 0.81),
        ]
    )
    output = build_model_comparison(baseline, torch_metrics)
    assert list(zip(output["split_protocol"], output["model"])) == [
        ("random", "RandomForestClassifier"),
        ("random", "TorchFingerprintMLP"),
        ("scaffold", "RandomForestClassifier"),
        ("scaffold", "TorchFingerprintMLP"),
    ]


def test_primary_model_selection_uses_scaffold_validation_not_test():
    baseline = pd.DataFrame(
        [
            _row("random", "RandomForestClassifier", 0.95, validation=0.80),
            _row("scaffold", "RandomForestClassifier", 0.99, validation=0.55),
        ]
    )
    torch_metrics = pd.DataFrame(
        [
            _row("random", "TorchFingerprintMLP", 0.70, validation=0.70),
            # Worse scaffold test metric, but better scaffold validation metric.
            _row("scaffold", "TorchFingerprintMLP", 0.60, validation=0.65),
        ]
    )
    selection = select_primary_scaffold_model(baseline, torch_metrics)
    selected = selection.loc[selection["selected_for_screen"], "model"].item()
    assert selected == "TorchFingerprintMLP"


def test_primary_model_selection_tie_prefers_simpler_random_forest():
    baseline = pd.DataFrame([_row("scaffold", "RandomForestClassifier", 0.5, validation=0.6)])
    torch_metrics = pd.DataFrame([_row("scaffold", "TorchFingerprintMLP", 0.9, validation=0.6)])
    selection = select_primary_scaffold_model(baseline, torch_metrics)
    selected = selection.loc[selection["selected_for_screen"], "model"].item()
    assert selected == "RandomForestClassifier"
