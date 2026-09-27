"""Build a small, reproducible ChEMBL screening-library snapshot.

This module deliberately does *not* query EGFR activity when constructing the
external screen. It obtains a target-agnostic pool from the ChEMBL molecule
endpoint, keeps records identified as small molecules with parseable canonical
SMILES, and then makes a deterministic hash-ranked slice.

The ChEMBL REST API is paginated. We request up to 1,000 records per page, check
that the live ChEMBL release matches the configured benchmark release, and
write both the ``.smi`` snapshot and a JSON manifest with its SHA-256 checksum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Iterable

import requests
from rdkit import Chem

from src.chembl import assert_expected_release, get_chembl_status, sha256_file

LOGGER = logging.getLogger(__name__)

CHEMBL_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
CHEMBL_HOMEPAGE = "https://www.ebi.ac.uk/chembl/"
CHEMBL_LICENSE = "Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)"


@dataclass(frozen=True)
class ScreeningLibrarySpec:
    """Parameters defining the frozen public-library slice."""

    expected_release: str = "ChEMBL_37"
    target_size: int = 30_000
    candidate_pool_size: int = 40_000
    page_size: int = 1_000
    seed: int = 2026
    refuse_release_mismatch: bool = True
    timeout_seconds: float = 60.0
    max_retries: int = 5

    def validate(self) -> None:
        if self.target_size <= 0:
            raise ValueError("target_size must be positive.")
        if self.candidate_pool_size < self.target_size:
            raise ValueError("candidate_pool_size must be >= target_size.")
        if not 1 <= self.page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000 for the ChEMBL API.")
        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1.")


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any],
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    """GET JSON with small, bounded retries for transient API failures."""

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                raise requests.HTTPError(
                    f"Transient ChEMBL HTTP {response.status_code}", response=response
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Unexpected ChEMBL response; expected a JSON object.")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            delay = min(2 ** (attempt - 1), 8)
            LOGGER.warning(
                "ChEMBL request failed (attempt %d/%d); retrying in %ds: %s",
                attempt,
                max_retries,
                delay,
                exc,
            )
            time.sleep(delay)
    assert last_error is not None
    raise RuntimeError("ChEMBL molecule request failed after retries.") from last_error


def _extract_small_molecule(record: dict[str, Any]) -> tuple[str, str] | None:
    """Return ``(canonical_smiles, molecule_chembl_id)`` for a usable record."""

    molecule_id = str(record.get("molecule_chembl_id") or "").strip()
    if not molecule_id:
        return None

    molecule_type = str(record.get("molecule_type") or "").strip().lower()
    if molecule_type and molecule_type != "small molecule":
        return None

    structures = record.get("molecule_structures")
    if not isinstance(structures, dict):
        return None
    smiles = str(structures.get("canonical_smiles") or "").strip()
    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return smiles, molecule_id


def _stable_key(molecule_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{molecule_id}".encode("utf-8")).hexdigest()


def deterministic_slice(
    records: Iterable[tuple[str, str]], *, target_size: int, seed: int
) -> list[tuple[str, str]]:
    """Select a stable target-agnostic slice, independent of input row order."""

    by_id: dict[str, str] = {}
    for smiles, molecule_id in records:
        by_id.setdefault(str(molecule_id), str(smiles))
    ranked = sorted(
        ((smiles, molecule_id) for molecule_id, smiles in by_id.items()),
        key=lambda item: (_stable_key(item[1], seed), item[1]),
    )
    if len(ranked) < target_size:
        raise ValueError(
            f"Only {len(ranked)} unique candidate molecules available for target_size={target_size}."
        )
    return ranked[:target_size]


def fetch_candidate_pool(
    spec: ScreeningLibrarySpec,
    *,
    session: requests.Session | None = None,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Fetch a target-agnostic pool of ChEMBL small molecules.

    Pagination is sequential and the final 30k subset is hash-ranked by ChEMBL
    molecule ID, which prevents the screen from being hand-selected for EGFR.
    """

    spec.validate()
    status = get_chembl_status(timeout=spec.timeout_seconds)
    if spec.refuse_release_mismatch:
        assert_expected_release(status, spec.expected_release)

    own_session = session is None
    session = session or requests.Session()
    candidates: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    offset = 0
    raw_records_scanned = 0
    pages_requested = 0

    try:
        while len(candidates) < spec.candidate_pool_size:
            payload = _request_json(
                session,
                CHEMBL_MOLECULE_URL,
                params={"limit": spec.page_size, "offset": offset},
                timeout=spec.timeout_seconds,
                max_retries=spec.max_retries,
            )
            pages_requested += 1
            molecules = payload.get("molecules")
            if not isinstance(molecules, list):
                raise ValueError("ChEMBL molecule response did not contain a 'molecules' list.")
            if not molecules:
                break

            raw_records_scanned += len(molecules)
            for record in molecules:
                if not isinstance(record, dict):
                    continue
                extracted = _extract_small_molecule(record)
                if extracted is None:
                    continue
                smiles, molecule_id = extracted
                if molecule_id in seen_ids:
                    continue
                seen_ids.add(molecule_id)
                candidates.append((smiles, molecule_id))
                if len(candidates) >= spec.candidate_pool_size:
                    break

            LOGGER.info(
                "screen_library_progress candidates=%d/%d raw_records=%d offset=%d",
                len(candidates),
                spec.candidate_pool_size,
                raw_records_scanned,
                offset,
            )

            page_meta = payload.get("page_meta")
            next_url = page_meta.get("next") if isinstance(page_meta, dict) else None
            if next_url in (None, "") and len(molecules) < spec.page_size:
                break
            offset += spec.page_size
    finally:
        if own_session:
            session.close()

    if len(candidates) < spec.candidate_pool_size:
        raise RuntimeError(
            "ChEMBL endpoint ended before the requested candidate pool was filled: "
            f"{len(candidates)} < {spec.candidate_pool_size}."
        )

    metadata = {
        "observed_chembl_release": status.get("chembl_db_version"),
        "observed_chembl_release_date": status.get("chembl_release_date"),
        "chembl_api_status": status.get("status"),
        "raw_records_scanned": int(raw_records_scanned),
        "pages_requested": int(pages_requested),
        "candidate_pool_count": int(len(candidates)),
    }
    return candidates, metadata


def write_screening_snapshot(
    selected: Iterable[tuple[str, str]],
    output_smi: str | Path,
    manifest_json: str | Path,
    *,
    spec: ScreeningLibrarySpec,
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write SMI + manifest and return the final manifest."""

    output_path = Path(output_smi)
    manifest_path = Path(manifest_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(selected)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# canonical_smiles molecule_chembl_id\n")
        for smiles, molecule_id in rows:
            handle.write(f"{smiles}\t{molecule_id}\n")

    retrieved = datetime.now(timezone.utc)
    release = source_metadata.get("observed_chembl_release") or spec.expected_release
    manifest: dict[str, Any] = {
        "name": f"{release} deterministic {spec.target_size:,}-molecule small-molecule slice",
        "source": CHEMBL_MOLECULE_URL,
        "source_homepage": CHEMBL_HOMEPAGE,
        "license": CHEMBL_LICENSE,
        "snapshot_date": retrieved.date().isoformat(),
        "retrieved_at_utc": retrieved.isoformat(),
        "selection": {
            "description": (
                "Target-agnostic ChEMBL molecule records; parseable small-molecule canonical "
                "SMILES collected into a fixed candidate pool, then ranked by "
                "SHA256(seed|molecule_chembl_id). No EGFR activity information is used."
            ),
            **asdict(spec),
        },
        "selected_count": int(len(rows)),
        "snapshot_path": str(output_path),
        "snapshot_sha256": sha256_file(output_path),
        **source_metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_screening_library(
    output_smi: str | Path,
    manifest_json: str | Path,
    *,
    spec: ScreeningLibrarySpec,
) -> dict[str, Any]:
    """Fetch, deterministically select, and freeze the screening library."""

    candidates, metadata = fetch_candidate_pool(spec)
    selected = deterministic_slice(candidates, target_size=spec.target_size, seed=spec.seed)
    return write_screening_snapshot(
        selected,
        output_smi,
        manifest_json,
        spec=spec,
        source_metadata=metadata,
    )


def load_library_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Screening-library manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Screening-library manifest must contain a JSON object.")
    return payload
