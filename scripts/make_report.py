"""Assemble model result tables and the two benchmark figures.

This stage requires completed baseline and Torch runs. It writes the held-out-test
comparison table, records the primary screening-model choice using scaffold
validation AUPRC only, and generates:

- results/figures/activity_distribution.png
- results/figures/scaffold_test_pr.png

Test metrics are never used to choose which model will rank the external library.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import feature_matrix_from_frame
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
    """Select the future screening model by scaffold-validation metric only."""

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


def plot_activity_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    """Plot exact pActivity measurements with the inactive/active cutoffs."""

    values = pd.to_numeric(frame["pactivity_median_exact"], errors="coerce").dropna().to_numpy()
    if values.size == 0:
        raise ValueError("No exact pActivity values are available for the activity-distribution figure.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(values, bins=45, edgecolor="black", linewidth=0.4)
    ax.axvline(5.0, linestyle="--", linewidth=1.4, label="inactive cutoff = 5.0")
    ax.axvline(6.0, linestyle="--", linewidth=1.4, label="active cutoff = 6.0")
    ax.set_xlabel("Median exact pActivity per standardized compound")
    ax.set_ylabel("Compounds")
    ax.set_title("EGFR activity distribution and classification cutoffs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scaffold_test_pr(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    model_path: Path,
    config: dict,
    output_path: Path,
) -> None:
    """Plot the scaffold-test precision-recall curve for the selected RF baseline."""

    if len(frame) != len(assignments):
        raise ValueError("Processed dataset and scaffold split assignment row counts differ.")
    if not frame["standardized_smiles"].astype(str).equals(
        assignments["standardized_smiles"].astype(str)
    ):
        raise ValueError("Processed dataset and scaffold split assignments are not row-aligned.")

    feature_cfg = config["features"]["morgan"]
    X = feature_matrix_from_frame(
        frame,
        radius=int(feature_cfg["radius"]),
        n_bits=int(feature_cfg["n_bits"]),
        use_chirality=bool(feature_cfg.get("use_chirality", True)),
    )
    test_mask = assignments["split"].astype(str).eq("test").to_numpy()
    y_test = frame.loc[test_mask, "activity_label"].astype(int).to_numpy()
    if len(np.unique(y_test)) < 2:
        raise ValueError("Scaffold test partition must contain both classes for a PR curve.")

    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    scores = model.predict_proba(X[test_mask])[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, scores)
    auprc = average_precision_score(y_test, scores)
    prevalence = float(np.mean(y_test))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(recall, precision, linewidth=1.8, label=f"Random Forest (AUPRC={auprc:.3f})")
    ax.axhline(prevalence, linestyle="--", linewidth=1.2, label=f"prevalence={prevalence:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Scaffold-test precision-recall curve")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = project_root_from_config_path(args.config)
    results_dir = resolve_path(root, config["paths"]["results_tables"])
    figures_dir = resolve_path(root, config["paths"]["results_figures"])
    models_dir = resolve_path(root, config["paths"]["results_models"])

    baseline_path = results_dir / "baseline_metrics.csv"
    torch_path = results_dir / "torch_metrics.csv"
    processed_path = resolve_path(root, config["data"]["chembl"]["processed_table"])
    assignment_path = results_dir / "scaffold_split_assignments.csv"
    scaffold_rf_path = models_dir / "scaffold_random_forest.pkl"

    required_paths = [baseline_path, torch_path, processed_path, assignment_path, scaffold_rf_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Report inputs are incomplete. Run data preparation and model stages first. Missing: "
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

    frame = pd.read_csv(processed_path)
    assignments = pd.read_csv(assignment_path)
    activity_figure = figures_dir / "activity_distribution.png"
    pr_figure = figures_dir / "scaffold_test_pr.png"
    plot_activity_distribution(frame, activity_figure)
    plot_scaffold_test_pr(frame, assignments, scaffold_rf_path, config, pr_figure)

    print(f"Wrote {comparison_path}")
    print(f"Wrote {selection_path}")
    print(f"Wrote {activity_figure}")
    print(f"Wrote {pr_figure}")


if __name__ == "__main__":
    main()
