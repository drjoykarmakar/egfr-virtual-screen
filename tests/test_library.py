import json
from pathlib import Path

from src.library import (
    ScreeningLibrarySpec,
    _extract_small_molecule,
    deterministic_slice,
    load_library_manifest,
    write_screening_snapshot,
)


def test_extract_small_molecule_requires_structure_and_small_molecule():
    good = {
        "molecule_chembl_id": "CHEMBL1",
        "molecule_type": "Small molecule",
        "molecule_structures": {"canonical_smiles": "CCO"},
    }
    assert _extract_small_molecule(good) == ("CCO", "CHEMBL1")

    non_small = dict(good, molecule_type="Protein")
    assert _extract_small_molecule(non_small) is None

    no_structure = dict(good, molecule_structures=None)
    assert _extract_small_molecule(no_structure) is None


def test_deterministic_slice_is_input_order_independent():
    records = [
        ("CCO", "CHEMBL1"),
        ("CCN", "CHEMBL2"),
        ("CCC", "CHEMBL3"),
        ("CCCl", "CHEMBL4"),
    ]
    a = deterministic_slice(records, target_size=3, seed=2026)
    b = deterministic_slice(list(reversed(records)), target_size=3, seed=2026)
    assert a == b
    assert len(a) == 3


def test_write_snapshot_records_required_provenance_and_checksum(tmp_path: Path):
    output = tmp_path / "screening_library.smi"
    manifest = tmp_path / "screening_library.manifest.json"
    spec = ScreeningLibrarySpec(target_size=2, candidate_pool_size=2, page_size=1000)
    payload = write_screening_snapshot(
        [("CCO", "CHEMBL1"), ("CCN", "CHEMBL2")],
        output,
        manifest,
        spec=spec,
        source_metadata={
            "observed_chembl_release": "ChEMBL_37",
            "observed_chembl_release_date": "2026-01-01",
            "chembl_api_status": "UP",
            "raw_records_scanned": 2,
            "pages_requested": 1,
            "candidate_pool_count": 2,
        },
    )
    assert output.exists()
    assert manifest.exists()
    assert payload["selected_count"] == 2
    assert payload["name"].startswith("ChEMBL_37")
    assert payload["source"].endswith("/molecule.json")
    assert "CC BY-SA 3.0" in payload["license"]
    assert len(payload["snapshot_sha256"]) == 64

    reloaded = load_library_manifest(manifest)
    assert reloaded["snapshot_sha256"] == payload["snapshot_sha256"]
    assert json.loads(manifest.read_text())["selection"]["seed"] == 2026
