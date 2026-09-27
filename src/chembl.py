"""ChEMBL retrieval helpers for the EGFR activity benchmark.

The important reproducibility rule is that the ChEMBL web API is mutable: it
serves the current database release, not an arbitrary historical release.
Therefore this module:

1. checks the API status endpoint and records the observed ChEMBL release;
2. optionally refuses to run when it differs from the configured release;
3. retrieves only human assays for the requested target;
4. retrieves IC50/Ki/Kd activities reported in nM;
5. writes an immutable-ish TSV snapshot plus a JSON manifest containing a
   SHA-256 checksum and the exact query settings.

Downstream modeling should use the frozen TSV referenced by the manifest rather
than silently re-querying the API on every run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd
import requests

CHEMBL_STATUS_URL = "https://www.ebi.ac.uk/chembl/api/data/status.json"


def _get_new_client():
    """Import the optional-at-runtime ChEMBL client only when retrieval is used."""

    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError as exc:
        raise RuntimeError(
            "chembl-webresource-client is required to download a fresh ChEMBL snapshot. "
            "Install requirements.txt, or reuse an existing frozen TSV snapshot."
        ) from exc
    return new_client


DEFAULT_FIELDS = [
    "activity_id",
    "assay_chembl_id",
    "molecule_chembl_id",
    "parent_molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "target_chembl_id",
    "target_organism",
    "target_pref_name",
    "document_chembl_id",
]


@dataclass(frozen=True)
class ChemblSnapshotSpec:
    """Configuration required to create a frozen ChEMBL activity snapshot."""

    target_chembl_id: str = "CHEMBL203"
    assay_organism: str = "Homo sapiens"
    standard_types: tuple[str, ...] = ("IC50", "Ki", "Kd")
    standard_units: str = "nM"
    allowed_relations: tuple[str, ...] = ("=", "<", "<=", ">", ">=")
    expected_release: str = "ChEMBL_37"
    refuse_release_mismatch: bool = True


def _normalise_release_name(value: str | None) -> str | None:
    if value is None:
        return None
    compact = str(value).strip().replace("-", "_")
    if compact.lower().startswith("chembl_"):
        suffix = compact.split("_", 1)[1]
        return f"ChEMBL_{suffix}"
    if compact.lower().startswith("chembl"):
        suffix = compact[6:].lstrip("_")
        return f"ChEMBL_{suffix}"
    return compact


def get_chembl_status(timeout: float = 30.0) -> dict:
    """Return ChEMBL API status metadata, including the database release.

    Raises:
        requests.HTTPError: if the status endpoint returns a non-success code.
        requests.RequestException: for network failures.
    """

    response = requests.get(CHEMBL_STATUS_URL, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Unexpected ChEMBL status payload; expected a JSON object.")
    return payload


def assert_expected_release(status: dict, expected_release: str) -> None:
    """Fail fast if the live ChEMBL API release is not the configured release."""

    observed = _normalise_release_name(status.get("chembl_db_version"))
    expected = _normalise_release_name(expected_release)
    if observed != expected:
        raise RuntimeError(
            "ChEMBL release mismatch: "
            f"expected {expected!r}, observed {observed!r}. "
            "Use a previously frozen snapshot or explicitly update the config "
            "after reviewing the new release."
        )


def _chunked(values: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def fetch_human_target_assays(
    target_chembl_id: str,
    assay_organism: str = "Homo sapiens",
) -> pd.DataFrame:
    """Fetch assays for one ChEMBL target and assay organism.

    Assay-organism filtering is done on the assay resource, not inferred from
    the target organism. This matters for experiments where assay and target
    organism metadata can differ.
    """

    fields = [
        "assay_chembl_id",
        "assay_organism",
        "assay_tax_id",
        "assay_type",
        "confidence_score",
        "target_chembl_id",
    ]
    new_client = _get_new_client()
    query = (
        new_client.assay.filter(target_chembl_id=target_chembl_id)
        .filter(assay_organism=assay_organism)
        .only(fields)
    )
    records = list(query)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError(
            f"No assays found for target={target_chembl_id} and "
            f"assay_organism={assay_organism!r}."
        )
    return frame.drop_duplicates(subset=["assay_chembl_id"]).reset_index(drop=True)


def fetch_activities_for_assays(
    assay_chembl_ids: Sequence[str],
    standard_types: Sequence[str] = ("IC50", "Ki", "Kd"),
    standard_units: str = "nM",
    allowed_relations: Sequence[str] = ("=", "<", "<=", ">", ">="),
    fields: Sequence[str] = DEFAULT_FIELDS,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """Fetch standardised activity rows for a collection of assay IDs."""

    new_client = _get_new_client()
    frames: list[pd.DataFrame] = []
    for assay_chunk in _chunked(list(assay_chembl_ids), chunk_size):
        query = (
            new_client.activity.filter(assay_chembl_id__in=assay_chunk)
            .filter(standard_type__in=list(standard_types))
            .filter(standard_units=standard_units)
            .filter(standard_value__isnull=False)
            .filter(standard_relation__in=list(allowed_relations))
            .only(list(fields))
        )
        records = list(query)
        if records:
            frames.append(pd.DataFrame.from_records(records))

    if not frames:
        columns = list(fields)
        return pd.DataFrame(columns=columns)

    result = pd.concat(frames, ignore_index=True)
    if "activity_id" in result.columns:
        result = result.drop_duplicates(subset=["activity_id"])
    return result.reset_index(drop=True)


def fetch_egfr_snapshot(spec: ChemblSnapshotSpec = ChemblSnapshotSpec()) -> tuple[pd.DataFrame, dict]:
    """Fetch a human EGFR activity snapshot and return data plus source metadata."""

    status = get_chembl_status()
    if spec.refuse_release_mismatch:
        assert_expected_release(status, spec.expected_release)

    assays = fetch_human_target_assays(
        target_chembl_id=spec.target_chembl_id,
        assay_organism=spec.assay_organism,
    )
    activities = fetch_activities_for_assays(
        assay_chembl_ids=assays["assay_chembl_id"].astype(str).tolist(),
        standard_types=spec.standard_types,
        standard_units=spec.standard_units,
        allowed_relations=spec.allowed_relations,
    )
    if activities.empty:
        raise RuntimeError("ChEMBL query returned no activity rows after filtering.")

    assay_meta = assays.rename(
        columns={
            "assay_organism": "source_assay_organism",
            "assay_tax_id": "source_assay_tax_id",
            "assay_type": "source_assay_type",
            "confidence_score": "source_assay_confidence_score",
        }
    )
    keep_meta = [
        c
        for c in [
            "assay_chembl_id",
            "source_assay_organism",
            "source_assay_tax_id",
            "source_assay_type",
            "source_assay_confidence_score",
        ]
        if c in assay_meta.columns
    ]
    activities = activities.merge(assay_meta[keep_meta], on="assay_chembl_id", how="left")

    metadata = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "observed_chembl_release": _normalise_release_name(status.get("chembl_db_version")),
        "observed_chembl_release_date": status.get("chembl_release_date"),
        "chembl_api_status": status.get("status"),
        "query": asdict(spec),
        "assay_count": int(len(assays)),
        "activity_row_count": int(len(activities)),
    }
    return activities, metadata


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_snapshot(
    activities: pd.DataFrame,
    metadata: dict,
    output_tsv: str | Path,
    manifest_json: str | Path,
) -> None:
    """Write a TSV snapshot and sidecar manifest with checksum."""

    output_tsv = Path(output_tsv)
    manifest_json = Path(manifest_json)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)

    activities.to_csv(output_tsv, sep="\t", index=False)
    manifest = dict(metadata)
    manifest.update(
        {
            "snapshot_path": str(output_tsv),
            "snapshot_sha256": sha256_file(output_tsv),
            "columns": list(activities.columns),
        }
    )
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def download_and_freeze(
    output_tsv: str | Path,
    manifest_json: str | Path,
    spec: ChemblSnapshotSpec = ChemblSnapshotSpec(),
) -> pd.DataFrame:
    """Convenience wrapper used by the data-preparation script."""

    activities, metadata = fetch_egfr_snapshot(spec=spec)
    write_snapshot(activities, metadata, output_tsv, manifest_json)
    return activities
