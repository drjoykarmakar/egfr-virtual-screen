# EGFR Virtual Screen

[![CI](https://github.com/YOUR_GITHUB_USERNAME/egfr-virtual-screen/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/egfr-virtual-screen/actions/workflows/ci.yml)

## Reviewer snapshot
- Target: human EGFR (CHEMBL203)
- Task: ligand-based virtual screen from ChEMBL bioactivity
- Stack: RDKit, scikit-learn, PyTorch
- Result: ranked public library plus nearest-training-active similarity
- Finding: TBD after scaffold-split evaluation and screening
- Not claimed: a new EGFR drug

## Scientific question

**If I train an activity model on cleaned EGFR ChEMBL data, how many of the highest-scoring molecules from a held-out public library are novel chemotypes rather than near-neighbor lookalikes of known actives?**

This repository is intentionally narrow. It treats a virtual-screen score as a ranking signal, then asks whether the ranking is doing more than retrieving analogs of molecules already present in the training set.

## Data source and activity cutoff

### EGFR bioactivity data

- Target: human EGFR / ErbB1, ChEMBL target ID `CHEMBL203`.
- ChEMBL release: expected `ChEMBL_37`; exact observed API release and release date are written to the snapshot manifest at download time.
- Retrieval date: `TBD`.
- Source: ChEMBL web services via `chembl_webresource_client`.
- Assay organism: `Homo sapiens`.
- Standard activity types kept: `IC50`, `Ki`, `Kd`.
- Standard unit required: `nM`.
- Snapshot files: `data/raw/egfr_chembl37_activities.tsv` and a sidecar JSON manifest containing the query settings, observed ChEMBL release, row count, and SHA-256 checksum.

The API reflects the current ChEMBL release rather than arbitrary historical releases, so `src/chembl.py` checks the API status endpoint before retrieval. By default it refuses to download if the observed release does not match `configs/default.yaml`. This prevents a future rerun from silently replacing the benchmark with a different ChEMBL release. For a fully frozen benchmark, keep the emitted snapshot file and manifest used for the reported results.

### Activity transformation and classification

For a measurement reported in nM:

```text
pActivity = 9 - log10(activity_nM)
```

Classification thresholds:

- active: `pActivity >= 6.0`, equivalent to `<= 1,000 nM`
- inactive / decoy-like: `pActivity <= 5.0`, equivalent to `>= 10,000 nM`
- gray zone: `5.0 < pActivity < 6.0`, excluded from classification training

Exact relations (`=`) are transformed directly. Censored relations are used only when the bound guarantees the class:

- `<` or `<=`: active only when the reported bound is already `<= 1,000 nM`
- `>` or `>=`: inactive only when the reported bound is already `>= 10,000 nM`
- all other censored measurements are left unlabeled for classification

A censored value is treated as a bound, not as an exact pActivity measurement.

### Duplicate policy

Rows are ultimately grouped by standardized parent SMILES.

- If a compound has both active and inactive labels after relation-aware labeling, the compound is dropped as conflicting.
- If labels agree, the compound is retained with that class label.
- Median pActivity is computed from exact (`=`) measurements only. Censored bounds are not inserted into the median as if they were exact observations.
- Counts of exact and censored supporting measurements are retained for auditability.

This is slightly stricter than taking the median of every numeric field because a numeric censoring limit is not an observed potency.

## Cleaning counts

The cleaning pipeline lives in `src/data.py` and logs counts after each transformation. Final numbers will be filled from the frozen EGFR snapshot.

| Step | Rows / compounds retained |
|---|---:|
| Raw ChEMBL activity rows | TBD |
| Parseable SMILES | TBD |
| Parent fragment obtained | TBD |
| Neutralization succeeded | TBD |
| Non-empty canonical isomeric SMILES | TBD |
| Relation-aware labeled rows | TBD |
| Gray-zone / ambiguous-censor rows removed from classification | TBD |
| Unique standardized compounds before conflict removal | TBD |
| Conflicting active/inactive compounds removed | TBD |
| Final classification compounds | TBD |
| Unique Bemis-Murcko scaffolds | TBD |

For each standardized molecule the processed table stores Bemis-Murcko scaffold, MW, cLogP, TPSA, HBD, HBA, rotatable bonds, Lipinski flag, Veber flag, QED, and PAINS status. A synthetic-accessibility score is included only when the lightweight RDKit `SA_Score` helper is importable; otherwise the column is left unavailable and the omission is reported.

## Methods

### Molecular representation

The baseline representation is Morgan radius 2, 2048 bits, concatenated with the same small RDKit descriptor set used for screening diagnostics:

- MW
- cLogP
- TPSA
- HBD
- HBA
- rotatable bonds
- QED

The Morgan bits stay binary. For the Torch model, only the seven continuous descriptor columns are standardized, using means and standard deviations fit on that protocol's training split only. Validation and test rows never contribute to scaling statistics. Because this is classification, the target is not normalized.

### Splits

Two evaluation protocols are reported:

1. **Random split** for an easier interpolation view.
2. **Bemis-Murcko scaffold split** for the primary estimate of generalization to new chemotypes.

Default fractions are 80% train, 10% validation, and 10% test. The scaffold splitter must place every Bemis-Murcko scaffold into exactly one split. The test suite asserts zero scaffold overlap across train, validation, and test for the scaffold protocol.

### Models

**Baseline:** `RandomForestClassifier` on Morgan fingerprints plus RDKit descriptors. Three small candidate settings are compared using validation AUPRC only; the test split is not consulted during model selection. The selected train-fit model is then evaluated once on the held-out test partition.

**Torch model:** a small CPU-runnable PyTorch MLP on the same fingerprint-plus-descriptor vector. Descriptor scaling is training-only, class imbalance is handled with a BCE positive-class weight computed from training labels only, and early stopping uses validation AUPRC. The best validation epoch is restored before the held-out test split is evaluated once. No language model, generative model, or foundation model is used.

The primary screening model is selected under the scaffold protocol. Random-split performance is context, not the headline result.

### Metrics

Classification metrics:

- AUROC
- AUPRC
- recall at top 1%
- recall at top 5%
- enrichment factor at 1% and 5% when the scaffold test set contains enough positives

The report also records the number of compounds and unique scaffolds in each split and verifies scaffold overlap is zero for the scaffold protocol.

## Model results on random vs scaffold split

Results are placeholders until the frozen dataset and both models have been run end to end.

| Split | Model | AUROC | AUPRC | Recall@1% | Recall@5% | EF@1% | EF@5% |
|---|---|---:|---:|---:|---:|---:|---:|
| Random | Random forest | TBD | TBD | TBD | TBD | TBD | TBD |
| Random | Torch MLP | TBD | TBD | TBD | TBD | TBD | TBD |
| Scaffold | Random forest | TBD | TBD | TBD | TBD | TBD | TBD |
| Scaffold | Torch MLP | TBD | TBD | TBD | TBD | TBD | TBD |

Scaffold counts:

| Split protocol | Train compounds / scaffolds | Validation compounds / scaffolds | Test compounds / scaffolds | Scaffold overlap |
|---|---:|---:|---:|---:|
| Random | TBD | TBD | TBD | expected, not constrained |
| Scaffold | TBD | TBD | TBD | **0 required** |

Primary model used for library ranking: **TBD after scaffold-validation comparison**. The selection rule is scaffold-validation AUPRC only; scaffold-test metrics are reported but are not used to choose the screening model. If validation AUPRC ties exactly, the simpler Random Forest is preferred.

Required scaffold-test figures:

- activity distribution with 5.0 / 6.0 pActivity cutoffs: `results/figures/activity_distribution.png` — TBD
- PR or ROC curve: `results/figures/scaffold_test_pr.png` — TBD

## Screen results: library size and top-20 table

Screening library: `TBD` public vendor-like / drug-like slice.

- Source: TBD
- License: TBD
- Download / snapshot date: TBD
- Raw library size: TBD
- Valid standardized molecules: TBD
- Exact standardized SMILES removed because they occur in EGFR train/validation/test: TBD
- Final molecules scored: TBD

The library is scored only after scaffold-split metrics exist. The primary scaffold-trained model ranks molecules by predicted active probability.

Cheap triage flags:

- MW between 200 and 600
- cLogP between -1 and 5
- QED >= 0.4
- PAINS reported as a separate flag rather than silently deleted

Nearest-known-active similarity is Morgan radius 2 Tanimoto against **actives in the scaffold-training partition only**. Validation/test actives are not used as the similarity reference. Exact standardized-SMILES overlap with the full EGFR benchmark (train, validation, and test) is removed before any external-library score is produced.

The external library may be supplied as `.smi`/`.smiles`, CSV, or TSV. Its name, source, license, and snapshot date must be filled in `configs/default.yaml`; `run_screen.py` refuses to score while those fields are still `TBD`. The screening summary records those fields plus a SHA-256 checksum of the actual input file.

Published output lists:

- `results/lists/top50_raw.csv`
- `results/lists/top50_filtered.csv`
- `results/lists/top20_interesting.csv`, restricted to molecules that pass the MW/cLogP/QED window and have nearest-training-active Tanimoto strictly `< 0.5`

For the interesting list, candidates are traversed in descending model-score order. Similarity evaluation stops once 20 qualifying molecules are found; if fewer than 20 qualify, the whole property-filtered library is examined and the smaller result is kept. The threshold is not relaxed after seeing the output.

Top-20 more-interesting molecules:

| Rank | Standardized SMILES | Active probability | QED | PAINS | Nearest training active | Tanimoto | Risk note |
|---:|---|---:|---:|---|---|---:|---|
| 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 6 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 7 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 9 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 10 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 11 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 12 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 13 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 14 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 15 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 17 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 18 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 19 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 20 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

The committed top-50 CSVs include at minimum standardized SMILES, model score, QED, PAINS flag, nearest scaffold-training active, Tanimoto similarity, and compact risk notes. `results/tables/screen_summary.json` records preparation counts, selected model family, similarity-reference size, filter counts, and top-score similarity statistics.

## How many top scores were near-neighbors of known actives

This is the central diagnostic, not a footnote.

- Top 50 raw with nearest-active Tanimoto `>= 0.5`: **TBD / 50**
- Top 50 raw with nearest-active Tanimoto `>= 0.7`: **TBD / 50**
- Top 200 score similarity median: **TBD**
- Top 200 score similarity IQR: **TBD**
- Molecules surviving all cheap filters and Tanimoto `< 0.5`: **TBD**

The most important figure is `results/figures/top200_nearest_active_similarity.png`.

If that histogram piles up near 0.7-1.0, the interpretation will be explicit: the screen is dominated by analog retrieval. A small `< 0.5` list will be reported as small rather than rescued by changing the threshold after seeing the results.

A molecule grid for the `< 0.5` shortlist will be saved as `results/figures/top_interesting_grid.png`.

## What failed

Populate this section with concrete negative results rather than deleting them from the history.

- ChEMBL records rejected by relation / unit / structure rules: TBD
- Activity conflicts after standardization: TBD
- Candidate model changes that did not improve scaffold-validation AUPRC: TBD
- Library molecules removed as exact training/evaluation duplicates: TBD
- Number of high-scoring molecules rejected by simple property filters: TBD
- Number of nominally high-scoring molecules that were close analogs of training actives: TBD
- Other failure or implementation note: TBD

## Limitations

- ChEMBL activity measurements combine assays with different protocols, contexts, and experimental uncertainty.
- IC50, Ki, and Kd are pooled for a deliberately simple portfolio benchmark; they are not physically interchangeable measurements.
- Binary cutoffs discard information and the 1-10 uM gray zone is intentionally omitted from classification.
- A scaffold split is harder than a random split but still does not reproduce prospective medicinal-chemistry deployment.
- Tanimoto novelty is fingerprint-dependent and is not a complete definition of chemotype novelty.
- PAINS alerts and simple property rules are triage flags, not proof that a molecule will or will not work experimentally.
- No selectivity, exposure, toxicity, permeability, metabolism, synthesis, crystal structure, or assay evidence is generated here.
- The external screening library is only a small public slice chosen to keep the workflow laptop-sized.

These compounds are computationally ranked hypotheses. They have not been synthesized or assayed in this repository.

## How to run

Python 3.11 is the reference environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

1. Freeze the expected ChEMBL release and download the EGFR activity snapshot:

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

2. Train and evaluate the baseline on random and scaffold splits:

```bash
python scripts/run_baseline.py --config configs/default.yaml
```

3. Train and evaluate the small PyTorch model:

```bash
python scripts/run_torch.py --config configs/default.yaml
```

4. Assemble the baseline-vs-Torch comparison and record the primary model selected from scaffold validation only:

```bash
python scripts/make_report.py --config configs/default.yaml
```

5. Build the frozen, target-agnostic 30,000-molecule ChEMBL screening-library slice:

```bash
python scripts/prepare_screening_library.py --config configs/default.yaml
```

The generator checks the live ChEMBL release against the configured release, pages through the public molecule API at up to 1,000 records per request, collects a fixed 40,000-record pool of parseable small molecules, then selects 30,000 by a deterministic SHA-256 ranking of `seed|molecule_chembl_id`. It does not query EGFR activity. It writes `data/raw/screening_library.smi` plus `data/raw/screening_library.manifest.json` with source, license, retrieval date, ChEMBL release, selection settings, and checksum. Re-running leaves an existing snapshot untouched.

6. Only after the model table, scaffold-validation selection record, and library manifest exist, score the external library:

```bash
python scripts/run_screen.py --config configs/default.yaml
```

`run_screen.py` reads provenance from the frozen manifest automatically, so no manual `TBD` editing is required. PAINS is hard-coded as `flag_only` for this benchmark; changing it to silent deletion is rejected.

The screen writes the three ranked CSVs, `results/tables/screen_summary.json`, the top-score similarity histogram, and a molecule-grid figure. If no molecule meets the filtered + `<0.5` criterion, the shortlist CSV is empty and the grid records that outcome rather than relaxing the rule.

Current implementation status:

- implemented: ChEMBL snapshot/reuse logic, activity labels, chemical standardization, descriptors/PAINS/QED, compound-level conflict handling
- implemented: Morgan + descriptor feature matrix
- implemented: reproducible stratified random split and label-blind Bemis-Murcko scaffold split with a zero-overlap assertion
- implemented: validation-only Random Forest tuning and test evaluation, with auditable split assignments
- implemented: small deterministic CPU PyTorch fingerprint MLP with training-only descriptor scaling, class weighting, early stopping, and reloadable checkpoints
- implemented: baseline-vs-Torch held-out-test comparison table plus scaffold-validation-only primary-model selection record
- implemented: deterministic ChEMBL 37 30k screening-library snapshot generator with release guard, manifest, checksum, and target-agnostic hash selection
- implemented: external-library SMI/CSV/TSV ingestion, shared standardization, exact full-benchmark overlap removal, validation-selected scaffold-model scoring, property filters, PAINS flagging, and scaffold-training-active Tanimoto triage
- implemented: top-50 raw/filtered lists, strict `<0.5` top-20 shortlist logic, screen audit JSON, nearest-active histogram, and shortlist molecule grid
- next: run the frozen 30k library screen, add the activity-distribution and scaffold-test PR figures, and populate README `TBD` values without changing thresholds post hoc

`scripts/run_baseline.py` writes `results/tables/baseline_metrics.csv`, `baseline_validation_candidates.csv`, `split_summary.csv`, and per-protocol split assignments. `scripts/run_torch.py` writes `torch_metrics.csv`, `torch_training_history.csv`, `torch_split_summary.csv`, and reloadable `.pt` checkpoints. `scripts/make_report.py` writes `model_comparison.csv` and `primary_model_selection.csv`. `scripts/run_screen.py` consumes the scaffold assignment and validation-selection record, then writes `screen_summary.json`, the three ranked lists, and the two screening figures. Generated model files under `results/models/` are local run artifacts and are not intended as evidence by themselves.

## What this repo does not claim

This repository does **not** present any molecule as an experimentally confirmed EGFR inhibitor, provide experimental validation, claim superiority over an industrial drug-discovery program, or identify a development candidate. It is not a de novo generation project, a multi-target platform, or a production-scale screen. Docking, if added later, will be an optional appendix and will not replace the ligand-based evidence or experimental validation.
