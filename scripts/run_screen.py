"""Score a frozen external library and perform skeptical nearest-active triage."""

from __future__ import annotations

import argparse
import hashlib
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

from src.features import feature_matrix_from_frame
from src.models_torch import load_torch_checkpoint, predict_active_probability
from src.library import load_library_manifest
from src.plots import plot_molecule_grid, plot_nearest_active_similarity
from src.screen import (
    NearestActiveIndex,
    annotate_nearest_active,
    annotate_risk_notes,
    apply_screen_filters,
    prepare_screening_library,
    read_screening_library,
    select_interesting_ranked,
    selected_model_name,
    training_actives_from_assignment,
)
from src.train import load_config, project_root_from_config_path, resolve_path

LOGGER = logging.getLogger("run_screen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def _resolve_library_provenance(library_cfg: dict, root: Path) -> dict[str, str]:
    """Resolve required provenance from the frozen library manifest."""

    manifest_value = library_cfg.get("manifest")
    if not manifest_value:
        raise ValueError("data.screening_library.manifest must be configured.")
    manifest_path = resolve_path(root, manifest_value)
    manifest = load_library_manifest(manifest_path)
    fields = ("name", "source", "license", "snapshot_date")
    resolved = {field: str(manifest.get(field, "")).strip() for field in fields}
    if bool(library_cfg.get("require_provenance", True)):
        missing = [field for field, value in resolved.items() if not value or value.upper() == "TBD"]
        if missing:
            raise ValueError(
                "Screening-library manifest is missing required provenance fields: "
                f"{missing}. Regenerate it with scripts/prepare_screening_library.py."
            )
    resolved["manifest_path"] = str(manifest_value)
    resolved["manifest_sha256"] = _sha256_file(manifest_path)
    return resolved


def _score_selected_model(
    model_name: str,
    X: np.ndarray,
    *,
    models_dir: Path,
    torch_batch_size: int,
    torch_device: str,
) -> np.ndarray:
    if model_name == "RandomForestClassifier":
        path = models_dir / "scaffold_random_forest.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Selected Random Forest checkpoint not found: {path}")
        with path.open("rb") as handle:
            model = pickle.load(handle)
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)

    if model_name == "TorchFingerprintMLP":
        path = models_dir / "scaffold_fingerprint_mlp.pt"
        if not path.exists():
            raise FileNotFoundError(f"Selected Torch checkpoint not found: {path}")
        model, standardizer, _ = load_torch_checkpoint(path, map_location=torch_device)
        return predict_active_probability(
            model,
            standardizer,
            X,
            batch_size=int(torch_batch_size),
            device=torch_device,
        ).astype(float)

    raise ValueError(f"Unsupported screening model: {model_name}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_columns(frame: pd.DataFrame, score_col: str) -> list[str]:
    preferred = [
        "rank",
        "interesting_rank",
        "library_id",
        "standardized_smiles",
        score_col,
        "mw",
        "clogp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "qed",
        "pains",
        "pains_description",
        "lipinski_ok",
        "veber_ok",
        "passes_mw",
        "passes_clogp",
        "passes_qed",
        "passes_property_filters",
        "nearest_training_active",
        "nearest_active_tanimoto",
        "risk_note",
    ]
    return [column for column in preferred if column in frame.columns]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    root = project_root_from_config_path(args.config)

    library_cfg = config["data"]["screening_library"]
    provenance = _resolve_library_provenance(library_cfg, root)

    benchmark_path = resolve_path(root, config["data"]["chembl"]["processed_table"])
    assignment_path = resolve_path(root, config["paths"]["results_tables"]) / "scaffold_split_assignments.csv"
    selection_path = resolve_path(root, config["paths"]["results_tables"]) / "primary_model_selection.csv"
    library_path = resolve_path(root, library_cfg["input_path"])
    required_paths = [benchmark_path, assignment_path, selection_path, library_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Screening prerequisites are incomplete. Run data/model stages and provide the "
            "library snapshot. Missing: " + ", ".join(missing)
        )

    benchmark = pd.read_csv(benchmark_path)
    assignment = pd.read_csv(assignment_path)
    selection = pd.read_csv(selection_path)
    model_name = selected_model_name(selection)

    raw_library = read_screening_library(
        library_path,
        smiles_column=library_cfg.get("smiles_column"),
        id_column=library_cfg.get("id_column"),
    )
    library, prep_report = prepare_screening_library(
        raw_library,
        benchmark["standardized_smiles"],
    )
    if library.empty:
        raise ValueError("No screening molecules remain after cleaning and benchmark-overlap removal.")

    target_size_min = int(library_cfg.get("target_size_min", 0))
    target_size_max = int(library_cfg.get("target_size_max", 10**9))
    if not target_size_min <= len(library) <= target_size_max:
        LOGGER.warning(
            "Final screening-library size %d is outside configured target range [%d, %d].",
            len(library),
            target_size_min,
            target_size_max,
        )

    feature_cfg = config["features"]["morgan"]
    X = feature_matrix_from_frame(
        library,
        radius=int(feature_cfg["radius"]),
        n_bits=int(feature_cfg["n_bits"]),
        use_chirality=bool(feature_cfg.get("use_chirality", True)),
    )
    models_dir = resolve_path(root, config["paths"]["results_models"])
    torch_cfg = config["torch"]
    scores = _score_selected_model(
        model_name,
        X,
        models_dir=models_dir,
        torch_batch_size=int(torch_cfg["batch_size"]),
        torch_device=str(torch_cfg.get("device", "cpu")),
    )

    score_col = str(config["screen"].get("score_column", "active_probability"))
    library[score_col] = scores
    filters = config["screen"]["filters"]
    if str(filters.get("pains_action", "flag_only")) != "flag_only":
        raise ValueError("This benchmark supports PAINS as flag_only; silent PAINS deletion is not allowed.")
    ranked = apply_screen_filters(
        library,
        mw_min=float(filters["mw_min"]),
        mw_max=float(filters["mw_max"]),
        clogp_min=float(filters["clogp_min"]),
        clogp_max=float(filters["clogp_max"]),
        qed_min=float(filters["qed_min"]),
    )
    ranked = ranked.sort_values(
        [score_col, "standardized_smiles"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))

    training_actives = training_actives_from_assignment(benchmark, assignment)
    nearest_cfg = config["screen"]["nearest_active"]
    similarity_index = NearestActiveIndex.from_smiles(
        training_actives["standardized_smiles"].astype(str).tolist(),
        radius=int(nearest_cfg["fingerprint_radius"]),
        n_bits=int(nearest_cfg["fingerprint_bits"]),
        use_chirality=bool(feature_cfg.get("use_chirality", True)),
    )

    sizes = config["screen"]["output_sizes"]
    raw_top_n = min(int(sizes["raw_top"]), len(ranked))
    histogram_top_n = min(int(sizes["similarity_histogram_top"]), len(ranked))
    filtered = ranked.loc[ranked["passes_property_filters"]].copy().reset_index(drop=True)
    filtered_top_n = min(int(sizes["filtered_top"]), len(filtered))

    # The top-200 annotation subsumes the top-50 raw list and drives the central histogram.
    top_for_similarity = annotate_nearest_active(
        ranked.head(histogram_top_n), similarity_index
    )
    top_for_similarity = annotate_risk_notes(top_for_similarity)
    top_raw = top_for_similarity.head(raw_top_n).copy()

    # Filtered top molecules can include rows outside the raw top-200, so annotate separately.
    top_filtered = annotate_nearest_active(filtered.head(filtered_top_n), similarity_index)
    top_filtered = annotate_risk_notes(top_filtered)

    interesting, interesting_examined = select_interesting_ranked(
        ranked,
        similarity_index,
        score_col=score_col,
        max_tanimoto=float(nearest_cfg["interesting_max_tanimoto"]),
        limit=int(sizes["interesting_top"]),
    )

    lists_dir = resolve_path(root, config["paths"]["results_lists"])
    figures_dir = resolve_path(root, config["paths"]["results_figures"])
    tables_dir = resolve_path(root, config["paths"]["results_tables"])
    for directory in (lists_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    top_raw.loc[:, _output_columns(top_raw, score_col)].to_csv(lists_dir / "top50_raw.csv", index=False)
    top_filtered.loc[:, _output_columns(top_filtered, score_col)].to_csv(
        lists_dir / "top50_filtered.csv", index=False
    )
    interesting.loc[:, _output_columns(interesting, score_col)].to_csv(
        lists_dir / "top20_interesting.csv", index=False
    )

    plot_nearest_active_similarity(
        top_for_similarity["nearest_active_tanimoto"],
        figures_dir / "top200_nearest_active_similarity.png",
    )
    plot_molecule_grid(
        interesting,
        figures_dir / "top_interesting_grid.png",
        score_col=score_col,
    )

    similarity_values = top_for_similarity["nearest_active_tanimoto"].astype(float)
    q1, median, q3 = np.quantile(similarity_values, [0.25, 0.50, 0.75])
    summary = {
        "library": {
            "name": provenance["name"],
            "source": provenance["source"],
            "license": provenance["license"],
            "snapshot_date": provenance["snapshot_date"],
            "manifest_path": provenance["manifest_path"],
            "manifest_sha256": provenance["manifest_sha256"],
            "input_path": str(library_cfg["input_path"]),
            "input_sha256": _sha256_file(library_path),
            **prep_report.to_dict(),
        },
        "model": {
            "selected_family": model_name,
            "selection_basis": "scaffold validation only",
            "score_column": score_col,
        },
        "nearest_active_reference": {
            "partition": "scaffold train",
            "n_training_actives": int(len(training_actives)),
            "morgan_radius": int(nearest_cfg["fingerprint_radius"]),
            "morgan_bits": int(nearest_cfg["fingerprint_bits"]),
        },
        "triage": {
            "top_raw_count": int(len(top_raw)),
            "top_filtered_count": int(len(top_filtered)),
            "interesting_count": int(len(interesting)),
            "interesting_candidates_examined": int(interesting_examined),
            "interesting_max_tanimoto_exclusive": float(nearest_cfg["interesting_max_tanimoto"]),
            "top_raw_tanimoto_ge_0_5": int(
                (top_raw["nearest_active_tanimoto"].astype(float) >= 0.5).sum()
            ),
            "top_raw_tanimoto_ge_0_7": int(
                (top_raw["nearest_active_tanimoto"].astype(float) >= 0.7).sum()
            ),
            "top_similarity_n": int(len(top_for_similarity)),
            "top_similarity_q1": float(q1),
            "top_similarity_median": float(median),
            "top_similarity_q3": float(q3),
            "top_similarity_iqr": float(q3 - q1),
            "property_filter_pass_count": int(ranked["passes_property_filters"].sum()),
            "property_filter_fail_count": int((~ranked["passes_property_filters"]).sum()),
            "pains_flag_count": int(ranked["pains"].sum()),
        },
    }
    summary_path = tables_dir / "screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    LOGGER.info("Selected scaffold model: %s", model_name)
    LOGGER.info("Scored %d non-overlapping standardized library molecules", len(ranked))
    LOGGER.info(
        "Interesting shortlist: %d molecules below Tanimoto %.2f after examining %d filtered candidates",
        len(interesting),
        float(nearest_cfg["interesting_max_tanimoto"]),
        interesting_examined,
    )
    LOGGER.info("Wrote screening outputs under %s, %s, and %s", lists_dir, figures_dir, tables_dir)


if __name__ == "__main__":
    main()
