"""Freeze ChEMBL EGFR activities and build the processed classification table."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chembl import ChemblSnapshotSpec, download_and_freeze
from src.data import (
    add_activity_columns,
    aggregate_classification_compounds,
    clean_smiles_dataframe,
)
from src.train import load_config, project_root_from_config_path, resolve_path

LOGGER = logging.getLogger("prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-query ChEMBL even if the configured frozen snapshot already exists.",
    )
    return parser.parse_args()


def _chembl_spec(config: dict) -> ChemblSnapshotSpec:
    chembl = config["data"]["chembl"]
    return ChemblSnapshotSpec(
        target_chembl_id=str(chembl["target_chembl_id"]),
        assay_organism=str(chembl["organism"]),
        standard_types=tuple(chembl["standard_types"]),
        standard_units=str(chembl["standard_units"]),
        allowed_relations=tuple(chembl["allowed_relations"]),
        expected_release=str(chembl["expected_release"]),
        refuse_release_mismatch=bool(chembl.get("refuse_release_mismatch", True)),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    root = project_root_from_config_path(args.config)
    chembl_cfg = config["data"]["chembl"]

    raw_path = resolve_path(root, chembl_cfg["raw_snapshot"])
    manifest_path = resolve_path(root, chembl_cfg["manifest"])
    processed_path = resolve_path(root, chembl_cfg["processed_table"])
    conflicts_path = resolve_path(root, chembl_cfg["conflicts_table"])
    report_path = resolve_path(root, chembl_cfg["cleaning_report"])

    if args.refresh or not raw_path.exists():
        LOGGER.info("Downloading ChEMBL snapshot to %s", raw_path)
        download_and_freeze(raw_path, manifest_path, spec=_chembl_spec(config))
    else:
        LOGGER.info("Reusing frozen ChEMBL snapshot %s", raw_path)

    raw = pd.read_csv(raw_path, sep="\t")
    labeled = add_activity_columns(
        raw,
        value_col="standard_value",
        relation_col="standard_relation",
        active_pactivity=float(chembl_cfg["active_pactivity"]),
        inactive_pactivity=float(chembl_cfg["inactive_pactivity"]),
    )
    cleaned, cleaning_report = clean_smiles_dataframe(labeled, smiles_col="canonical_smiles")
    retained, conflicts = aggregate_classification_compounds(cleaned)

    # Chemical properties are deterministic for a standardized SMILES, so one
    # representative row is sufficient when merging them onto aggregated labels.
    descriptor_columns = [
        "standardized_smiles",
        "scaffold_smiles",
        "mw",
        "clogp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "lipinski_ok",
        "veber_ok",
        "qed",
        "pains",
        "pains_description",
        "sa_score",
        "sa_score_available",
    ]
    descriptors = (
        cleaned.loc[:, descriptor_columns]
        .drop_duplicates(subset=["standardized_smiles"], keep="first")
        .reset_index(drop=True)
    )
    processed = retained.merge(descriptors, on="standardized_smiles", how="left", validate="one_to_one")
    processed = processed.sort_values("standardized_smiles").reset_index(drop=True)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    conflicts_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_path, index=False)
    conflicts.to_csv(conflicts_path, index=False)

    report = {
        "cleaning": cleaning_report.to_dict(),
        "raw_activity_rows": int(len(raw)),
        "relation_aware_labeled_rows": int(cleaned["activity_label"].notna().sum()),
        "classification_excluded_rows": int(cleaned["activity_label"].isna().sum()),
        "unique_standardized_compounds_before_conflict_removal": int(
            cleaned.loc[cleaned["activity_label"].notna(), "standardized_smiles"].nunique()
        ),
        "conflicting_compounds_removed": int(len(conflicts)),
        "final_classification_compounds": int(len(processed)),
        "final_unique_scaffolds": int(processed["scaffold_smiles"].nunique(dropna=False)),
        "final_actives": int((processed["activity_label"].astype(int) == 1).sum()),
        "final_inactives": int((processed["activity_label"].astype(int) == 0).sum()),
        "sa_score_available_for_all": bool(processed["sa_score_available"].all()) if len(processed) else False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    LOGGER.info("Wrote %d classification compounds to %s", len(processed), processed_path)
    LOGGER.info("Dropped %d compounds with conflicting labels", len(conflicts))
    LOGGER.info("Wrote cleaning report to %s", report_path)


if __name__ == "__main__":
    main()
