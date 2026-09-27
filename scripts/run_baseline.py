"""Train/evaluate the Random Forest baseline under random and scaffold splits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import classification_metrics
from src.features import feature_matrix_from_frame
from src.models_baseline import fit_baseline_with_validation
from src.splits import (
    assert_zero_scaffold_overlap,
    random_split_indices,
    scaffold_split_indices,
    split_assignment_frame,
    split_summary,
)
from src.train import load_config, project_root_from_config_path, resolve_path

LOGGER = logging.getLogger("run_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def _make_split(frame: pd.DataFrame, protocol: str, config: dict):
    split_cfg = config["splits"]
    kwargs = {
        "train_fraction": float(split_cfg["train_fraction"]),
        "validation_fraction": float(split_cfg["validation_fraction"]),
        "test_fraction": float(split_cfg["test_fraction"]),
    }
    if protocol == "random":
        return random_split_indices(
            frame["activity_label"].astype(int).to_numpy(),
            seed=int(split_cfg["random_seed"]),
            stratify=True,
            **kwargs,
        )
    if protocol == "scaffold":
        return scaffold_split_indices(frame["scaffold_smiles"].fillna("").astype(str), **kwargs)
    raise ValueError(f"Unknown split protocol: {protocol}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    root = project_root_from_config_path(args.config)

    processed_path = resolve_path(root, config["data"]["chembl"]["processed_table"])
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {processed_path}. Run scripts/prepare_data.py first."
        )
    frame = pd.read_csv(processed_path)
    if frame.empty:
        raise ValueError("Processed classification table is empty.")
    frame["activity_label"] = frame["activity_label"].astype(int)

    feature_cfg = config["features"]["morgan"]
    X = feature_matrix_from_frame(
        frame,
        radius=int(feature_cfg["radius"]),
        n_bits=int(feature_cfg["n_bits"]),
        use_chirality=bool(feature_cfg.get("use_chirality", True)),
    )
    y = frame["activity_label"].to_numpy(dtype=int)

    baseline_cfg = config["baseline"]
    candidates = list(baseline_cfg["validation_candidates"])
    results_dir = resolve_path(root, config["paths"]["results_tables"])
    models_dir = resolve_path(root, config["paths"]["results_models"])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    candidate_rows: list[dict] = []
    summary_rows: list[pd.DataFrame] = []

    for protocol in ("random", "scaffold"):
        split = _make_split(frame, protocol, config)
        if protocol == "scaffold":
            assert_zero_scaffold_overlap(frame["scaffold_smiles"].fillna("").astype(str), split)

        fit = fit_baseline_with_validation(
            X[split.train],
            y[split.train],
            X[split.validation],
            y[split.validation],
            candidates,
            random_state=int(baseline_cfg["random_state"]),
            class_weight=baseline_cfg.get("class_weight", "balanced"),
            n_jobs=int(baseline_cfg.get("n_jobs", -1)),
            selection_metric=str(baseline_cfg.get("selection_metric", "auprc")),
        )
        test_scores = fit.model.predict_proba(X[split.test])[:, 1]
        test_metrics = classification_metrics(y[split.test], test_scores)
        metric_rows.append(
            {
                "split_protocol": protocol,
                "model": "RandomForestClassifier",
                "n_train": int(len(split.train)),
                "n_validation": int(len(split.validation)),
                "n_test": int(len(split.test)),
                **test_metrics,
                "validation_selection_metric": float(
                    fit.validation_metrics[str(baseline_cfg.get("selection_metric", "auprc"))]
                ),
                "selection_metric": str(baseline_cfg.get("selection_metric", "auprc")),
                "best_params": json.dumps(fit.best_params, sort_keys=True),
            }
        )
        for row in fit.candidate_results:
            candidate_rows.append({"split_protocol": protocol, **row})

        summary = split_summary(frame, split)
        summary.insert(0, "split_protocol", protocol)
        summary_rows.append(summary)

        assignment = split_assignment_frame(frame, split)
        assignment.to_csv(results_dir / f"{protocol}_split_assignments.csv", index=False)
        with (models_dir / f"{protocol}_random_forest.pkl").open("wb") as handle:
            pickle.dump(fit.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        LOGGER.info(
            "%s baseline test: AUROC=%s AUPRC=%s",
            protocol,
            f"{test_metrics['auroc']:.3f}" if np.isfinite(test_metrics["auroc"]) else "NaN",
            f"{test_metrics['auprc']:.3f}" if np.isfinite(test_metrics["auprc"]) else "NaN",
        )

    pd.DataFrame(metric_rows).to_csv(results_dir / "baseline_metrics.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(results_dir / "baseline_validation_candidates.csv", index=False)
    pd.concat(summary_rows, ignore_index=True).to_csv(results_dir / "split_summary.csv", index=False)
    LOGGER.info("Wrote baseline metrics and auditable split assignments under %s", results_dir)


if __name__ == "__main__":
    main()
