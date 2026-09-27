# Data policy

Large mutable source files are not committed by default.

- `raw/`: frozen source snapshots and JSON manifests. The default `.gitignore` keeps large raw tables out of git; publish a stable download reference or attach a release asset when the benchmark is finalized.
- `processed/`: reproducible derived tables created by the preparation scripts.

For ChEMBL, always retain the snapshot manifest with observed release, retrieval time, query settings, and SHA-256 checksum.

## Processed outputs

`python scripts/prepare_data.py --config configs/default.yaml` creates:

- `data/processed/egfr_activity_classification.csv`: one retained classification row per standardized parent SMILES
- `data/processed/egfr_activity_conflicts.csv`: compounds dropped because both active and inactive labels were observed
- `data/processed/egfr_cleaning_report.json`: cleaning and class-count audit trail

These processed files are reproducible run artifacts and are gitignored by default.

## Screening library snapshot

The default workflow creates its own laptop-sized, target-agnostic ChEMBL slice:

```bash
python scripts/prepare_screening_library.py --config configs/default.yaml
```

The generator checks the live ChEMBL release, requests the public molecule endpoint in pages of up to 1,000 records, keeps parseable small-molecule canonical SMILES until a fixed candidate pool is filled, and deterministically selects 30,000 molecules by SHA-256 ranking of `seed|molecule_chembl_id`. No EGFR target/activity information is used to choose the library.

It writes:

- `data/raw/screening_library.smi`: canonical SMILES plus ChEMBL molecule ID
- `data/raw/screening_library.manifest.json`: ChEMBL release/date, API source, CC BY-SA 3.0 provenance, selection parameters, row counts, and SHA-256 checksum

The raw `.smi` stays gitignored; the small JSON manifest may be committed. An existing snapshot+manifest is left unchanged unless the user intentionally deletes both before regeneration.

`run_screen.py` reads required provenance from the manifest, standardizes the library with the same molecule-cleaning code used for EGFR, collapses duplicate standardized SMILES, removes exact standardized-SMILES overlap with the entire processed EGFR benchmark, and records the input SHA-256 plus all preparation counts in `results/tables/screen_summary.json`.

The generic screening reader still supports manually supplied `.smi`/`.smiles`, CSV, or TSV files if the config path/manifest are changed deliberately.
