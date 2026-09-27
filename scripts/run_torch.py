"""Train/evaluate the small fingerprint MLP under random and scaffold splits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import classification_metrics
from src.features import feature_matrix_from_frame
from src.models_torch import (
    fit_fingerprint_mlp,
    predict_active_probability,
    save_torch_checkpoint,
)
from src.splits import (
    assert_zero_scaffold_overlap,
    random_split_indices,
    scaffold_split_indices,
    split_assignment_frame,
    split_summary,
)
from src.train import load_config, project_root_from_config_path, resolve_path

LOGGER = logging.getLogger("run_torch")


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
    n_fingerprint_features = int(feature_cfg["n_bits"])
    X = feature_matrix_from_frame(
        frame,
        radius=int(feature_cfg["radius"]),
        n_bits=n_fingerprint_features,
        use_chirality=bool(feature_cfg.get("use_chirality", True)),
    )
    y = frame["activity_label"].to_numpy(dtype=int)

    torch_cfg = config["torch"]
    results_dir = resolve_path(root, config["paths"]["results_tables"])
    models_dir = resolve_path(root, config["paths"]["results_models"])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    history_rows: list[pd.DataFrame] = []
    summary_rows: list[pd.DataFrame] = []

    for protocol in ("random", "scaffold"):
        split = _make_split(frame, protocol, config)
        if protocol == "scaffold":
            assert_zero_scaffold_overlap(frame["scaffold_smiles"].fillna("").astype(str), split)

        fit = fit_fingerprint_mlp(
            X[split.train],
            y[split.train],
            X[split.validation],
            y[split.validation],
            n_fingerprint_features=n_fingerprint_features,
            hidden_dims=[int(value) for value in torch_cfg["hidden_dims"]],
            dropout=float(torch_cfg["dropout"]),
            batch_size=int(torch_cfg["batch_size"]),
            learning_rate=float(torch_cfg["learning_rate"]),
            weight_decay=float(torch_cfg["weight_decay"]),
            max_epochs=int(torch_cfg["max_epochs"]),
            early_stopping_patience=int(torch_cfg["early_stopping_patience"]),
            selection_metric=str(torch_cfg.get("selection_metric", "auprc")),
            device=str(torch_cfg.get("device", "cpu")),
            seed=int(torch_cfg["seed"]),
        )
        test_scores = predict_active_probability(
            fit.model,
            fit.standardizer,
            X[split.test],
            batch_size=int(torch_cfg["batch_size"]),
            device=str(torch_cfg.get("device", "cpu")),
        )
        test_metrics = classification_metrics(y[split.test], test_scores)
        metric_rows.append(
            {
                "split_protocol": protocol,
                "model": "TorchFingerprintMLP",
                "n_train": int(len(split.train)),
                "n_validation": int(len(split.validation)),
                "n_test": int(len(split.test)),
                **test_metrics,
                "best_epoch": int(fit.best_epoch),
                "validation_selection_metric": float(
                    fit.validation_metrics[str(torch_cfg.get("selection_metric", "auprc"))]
                ),
                "selection_metric": str(torch_cfg.get("selection_metric", "auprc")),
                "pos_weight": float(fit.pos_weight),
                "hidden_dims": json.dumps([int(value) for value in torch_cfg["hidden_dims"]]),
                "dropout": float(torch_cfg["dropout"]),
            }
        )

        history = pd.DataFrame(fit.history)
        history.insert(0, "split_protocol", protocol)
        history_rows.append(history)

        summary = split_summary(frame, split)
        summary.insert(0, "split_protocol", protocol)
        summary_rows.append(summary)

        # Rewriting the same deterministic assignment is intentional: it makes
        # the apples-to-apples split used by both model families independently auditable.
        assignment = split_assignment_frame(frame, split)
        assignment.to_csv(results_dir / f"{protocol}_split_assignments.csv", index=False)

        checkpoint_path = models_dir / f"{protocol}_fingerprint_mlp.pt"
        save_torch_checkpoint(
            checkpoint_path,
            fit,
            feature_config={
                "radius": int(feature_cfg["radius"]),
                "n_bits": n_fingerprint_features,
                "use_chirality": bool(feature_cfg.get("use_chirality", True)),
            },
        )

        LOGGER.info(
            "%s torch test: AUROC=%s AUPRC=%s (best epoch %d)",
            protocol,
            f"{test_metrics['auroc']:.3f}" if np.isfinite(test_metrics["auroc"]) else "NaN",
            f"{test_metrics['auprc']:.3f}" if np.isfinite(test_metrics["auprc"]) else "NaN",
            fit.best_epoch,
        )

    pd.DataFrame(metric_rows).to_csv(results_dir / "torch_metrics.csv", index=False)
    pd.concat(history_rows, ignore_index=True).to_csv(
        results_dir / "torch_training_history.csv", index=False
    )
    pd.concat(summary_rows, ignore_index=True).to_csv(
        results_dir / "torch_split_summary.csv", index=False
    )
    LOGGER.info("Wrote Torch metrics to %s and checkpoints to %s", results_dir, models_dir)


if __name__ == "__main__":
    main()
