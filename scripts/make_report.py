"""Assemble model result tables from completed baseline and Torch runs.

This stage intentionally does not fabricate missing results. It requires both
metric files, writes the four-row held-out-test comparison, and records the
primary screening-model choice using scaffold *validation* AUPRC only. Test
metrics are never used to choose which model will rank the external library.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import load_config, project_root_from_config_path, resolve_path


METRIC_COLUMNS = [
    "auroc",
    "auprc",
    "recall_at_1pct",
    "recall_at_5pct",
    "ef_at_1pct",
    "ef_at_5pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def build_model_comparison(baseline: pd.DataFrame, torch_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return held-out-test metrics in stable README order."""

    required = {"split_protocol", "model", *METRIC_COLUMNS}
    for name, frame in (("baseline", baseline), ("torch", torch_metrics)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} metrics are missing required columns: {missing}")

    combined = pd.concat([baseline, torch_metrics], ignore_index=True, sort=False)
    expected_pairs = {
        ("random", "RandomForestClassifier"),
        ("scaffold", "RandomForestClassifier"),
        ("random", "TorchFingerprintMLP"),
        ("scaffold", "TorchFingerprintMLP"),
    }
    observed_pairs = set(zip(combined["split_protocol"], combined["model"], strict=False))
    missing_pairs = sorted(expected_pairs.difference(observed_pairs))
    if missing_pairs:
        raise ValueError(f"Missing expected model/split results: {missing_pairs}")

    split_order = pd.Categorical(
        combined["split_protocol"], categories=["random", "scaffold"], ordered=True
    )
    model_order = pd.Categorical(
        combined["model"],
        categories=["RandomForestClassifier", "TorchFingerprintMLP"],
        ordered=True,
    )
    output = combined.assign(_split_order=split_order, _model_order=model_order).sort_values(
        ["_split_order", "_model_order"]
    )
    keep = ["split_protocol", "model", *METRIC_COLUMNS]
    return output.loc[:, keep].reset_index(drop=True)


def select_primary_scaffold_model(
    baseline: pd.DataFrame,
    torch_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Select the future screening model by scaffold-validation metric only.

    A deterministic simplicity tie-break favors the Random Forest if validation
    scores are numerically equal. The returned table is an audit record; test
    metrics are deliberately absent.
    """

    required = {"split_protocol", "model", "validation_selection_metric", "selection_metric"}
    candidates = pd.concat([baseline, torch_metrics], ignore_index=True, sort=False)
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Model metrics are missing selection columns: {missing}")

    scaffold = candidates.loc[candidates["split_protocol"] == "scaffold"].copy()
    expected_models = {"RandomForestClassifier", "TorchFingerprintMLP"}
    observed_models = set(scaffold["model"])
    if not expected_models.issubset(observed_models):
        raise ValueError(
            "Scaffold-validation selection requires both model families; missing "
            f"{sorted(expected_models.difference(observed_models))}."
        )
    metrics = set(scaffold["selection_metric"].astype(str))
    if len(metrics) != 1:
        raise ValueError(f"Model families used different selection metrics: {sorted(metrics)}")
    values = scaffold["validation_selection_metric"].astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Scaffold validation selection metric contains NaN or infinity.")

    tie_priority = {"RandomForestClassifier": 0, "TorchFingerprintMLP": 1}
    scaffold["_tie_priority"] = scaffold["model"].map(tie_priority).fillna(99)
    scaffold = scaffold.sort_values(
        ["validation_selection_metric", "_tie_priority"],
        ascending=[False, True],
    ).reset_index(drop=True)
    scaffold["selected_for_screen"] = False
    scaffold.loc[0, "selected_for_screen"] = True
    return scaffold.loc[
        :, ["model", "selection_metric", "validation_selection_metric", "selected_for_screen"]
    ]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = project_root_from_config_path(args.config)
    results_dir = resolve_path(root, config["paths"]["results_tables"])
    baseline_path = results_dir / "baseline_metrics.csv"
    torch_path = results_dir / "torch_metrics.csv"
    missing = [str(path) for path in (baseline_path, torch_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Model metrics are incomplete. Run baseline and Torch stages first. Missing: "
            + ", ".join(missing)
        )

    baseline = pd.read_csv(baseline_path)
    torch_metrics = pd.read_csv(torch_path)
    comparison = build_model_comparison(baseline, torch_metrics)
    selection = select_primary_scaffold_model(baseline, torch_metrics)

    comparison_path = results_dir / "model_comparison.csv"
    selection_path = results_dir / "primary_model_selection.csv"
    comparison.to_csv(comparison_path, index=False)
    selection.to_csv(selection_path, index=False)
    print(f"Wrote {comparison_path}")
    print(f"Wrote {selection_path}")


if __name__ == "__main__":
    main()
