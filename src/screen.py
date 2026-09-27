"""External-library scoring and skeptical nearest-active triage.

The screening stage is intentionally conservative:

* external molecules are standardized with the same chemistry rules as ChEMBL;
* exact standardized-SMILES overlap with the full EGFR benchmark is removed;
* only the model selected from scaffold-validation evidence is allowed to score;
* PAINS is reported as a flag and is never silently deleted;
* "interesting" means property-filter-passing and Tanimoto < 0.5 to every
  *training-set* active under the scaffold protocol.

A high score is therefore a ranking hypothesis, not an activity claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.data import CleaningReport, clean_smiles_dataframe


SMILES_COLUMN_CANDIDATES: tuple[str, ...] = (
    "smiles",
    "SMILES",
    "canonical_smiles",
    "standardized_smiles",
)
ID_COLUMN_CANDIDATES: tuple[str, ...] = (
    "library_id",
    "compound_id",
    "molecule_id",
    "id",
    "ID",
    "name",
)


@dataclass(frozen=True)
class LibraryPreparationReport:
    """Counts needed to audit external-library preparation."""

    raw_rows: int
    parsed_smiles: int
    parent_fragment: int
    neutralized: int
    canonical_nonempty: int
    standardized_rows: int
    standardized_duplicates_removed: int
    exact_benchmark_overlaps_removed: int
    final_rows: int

    def to_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class NearestActiveIndex:
    """Morgan fingerprints for exact nearest-neighbor lookup."""

    smiles: tuple[str, ...]
    fingerprints: tuple[DataStructs.ExplicitBitVect, ...]
    radius: int
    n_bits: int
    use_chirality: bool

    @classmethod
    def from_smiles(
        cls,
        smiles: Sequence[str],
        *,
        radius: int = 2,
        n_bits: int = 2048,
        use_chirality: bool = True,
    ) -> "NearestActiveIndex":
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=int(radius),
            fpSize=int(n_bits),
            includeChirality=bool(use_chirality),
        )
        kept_smiles: list[str] = []
        fingerprints: list[DataStructs.ExplicitBitVect] = []
        for value in smiles:
            text = str(value)
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                raise ValueError(f"Invalid standardized active SMILES: {text!r}")
            kept_smiles.append(text)
            fingerprints.append(generator.GetFingerprint(mol))
        if not fingerprints:
            raise ValueError("At least one training active is required for nearest-active similarity.")
        return cls(
            smiles=tuple(kept_smiles),
            fingerprints=tuple(fingerprints),
            radius=int(radius),
            n_bits=int(n_bits),
            use_chirality=bool(use_chirality),
        )

    def nearest(self, smiles: str) -> tuple[str, float]:
        """Return nearest training-active SMILES and exact Tanimoto similarity."""

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Invalid standardized query SMILES: {smiles!r}")
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits,
            includeChirality=self.use_chirality,
        )
        query = generator.GetFingerprint(mol)
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(query, list(self.fingerprints)),
            dtype=float,
        )
        best_index = int(np.argmax(similarities))
        return self.smiles[best_index], float(similarities[best_index])


def _first_present(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def read_screening_library(
    path: str | Path,
    *,
    smiles_column: str | None = None,
    id_column: str | None = None,
) -> pd.DataFrame:
    """Read a small public screening library from SMI, CSV, or TSV.

    ``.smi``/``.smiles`` files are interpreted as whitespace-separated files in
    which the first token is SMILES and the optional second token is an ID.
    CSV/TSV inputs may specify column names explicitly; otherwise common SMILES
    and identifier names are detected.
    """

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Screening library not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix in {".smi", ".smiles"}:
        records: list[dict[str, str]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                fields = stripped.split()
                if not fields:
                    continue
                records.append(
                    {
                        "input_smiles": fields[0],
                        "library_id": fields[1] if len(fields) > 1 else f"row_{line_number}",
                    }
                )
        return pd.DataFrame.from_records(records, columns=["input_smiles", "library_id"])

    if suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(input_path, sep="\t")
    elif suffix == ".csv":
        frame = pd.read_csv(input_path)
    else:
        raise ValueError(
            f"Unsupported screening-library format {suffix!r}; use .smi, .smiles, .csv, or .tsv."
        )

    resolved_smiles = smiles_column or _first_present(frame.columns, SMILES_COLUMN_CANDIDATES)
    if resolved_smiles is None or resolved_smiles not in frame.columns:
        raise KeyError(
            "Could not identify a SMILES column. Configure data.screening_library.smiles_column."
        )
    resolved_id = id_column or _first_present(frame.columns, ID_COLUMN_CANDIDATES)

    output = pd.DataFrame({"input_smiles": frame[resolved_smiles]})
    if resolved_id is not None and resolved_id in frame.columns:
        output["library_id"] = frame[resolved_id].astype(str)
    else:
        output["library_id"] = [f"row_{index + 1}" for index in range(len(frame))]
    return output


def prepare_screening_library(
    raw_library: pd.DataFrame,
    benchmark_smiles: Iterable[str],
    *,
    smiles_col: str = "input_smiles",
) -> tuple[pd.DataFrame, LibraryPreparationReport]:
    """Standardize, deduplicate, and remove exact benchmark overlap.

    Exact overlap is checked against the full processed EGFR benchmark (train,
    validation, and test), not merely the scaffold-training partition.
    """

    if smiles_col not in raw_library.columns:
        raise KeyError(f"Missing screening-library SMILES column: {smiles_col!r}")

    cleaned, cleaning = clean_smiles_dataframe(raw_library, smiles_col=smiles_col, log=None)
    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["standardized_smiles"], keep="first").copy()
    duplicates_removed = before_dedup - len(cleaned)

    benchmark = {str(value) for value in benchmark_smiles if pd.notna(value)}
    overlap_mask = cleaned["standardized_smiles"].astype(str).isin(benchmark)
    overlaps_removed = int(overlap_mask.sum())
    cleaned = cleaned.loc[~overlap_mask].copy()
    cleaned = cleaned.sort_values(["standardized_smiles", "library_id"], kind="stable").reset_index(drop=True)

    report = LibraryPreparationReport(
        raw_rows=int(len(raw_library)),
        parsed_smiles=int(cleaning.parsed_smiles),
        parent_fragment=int(cleaning.parent_fragment),
        neutralized=int(cleaning.neutralized),
        canonical_nonempty=int(cleaning.canonical_nonempty),
        standardized_rows=int(before_dedup),
        standardized_duplicates_removed=int(duplicates_removed),
        exact_benchmark_overlaps_removed=int(overlaps_removed),
        final_rows=int(len(cleaned)),
    )
    return cleaned, report


def apply_screen_filters(
    frame: pd.DataFrame,
    *,
    mw_min: float = 200.0,
    mw_max: float = 600.0,
    clogp_min: float = -1.0,
    clogp_max: float = 5.0,
    qed_min: float = 0.40,
) -> pd.DataFrame:
    """Add transparent property-filter columns without deleting PAINS alerts."""

    required = {"mw", "clogp", "qed", "pains"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Screening frame is missing filter columns: {missing}")
    if mw_min > mw_max or clogp_min > clogp_max:
        raise ValueError("Filter lower bounds must not exceed upper bounds.")

    out = frame.copy()
    out["passes_mw"] = out["mw"].between(float(mw_min), float(mw_max), inclusive="both")
    out["passes_clogp"] = out["clogp"].between(
        float(clogp_min), float(clogp_max), inclusive="both"
    )
    out["passes_qed"] = pd.to_numeric(out["qed"], errors="coerce").ge(float(qed_min))
    out["passes_property_filters"] = out[["passes_mw", "passes_clogp", "passes_qed"]].all(axis=1)
    # PAINS is intentionally separate. It does not participate in the pass/fail mask.
    out["pains"] = out["pains"].fillna(False).astype(bool)
    return out


def training_actives_from_assignment(
    benchmark: pd.DataFrame,
    scaffold_assignment: pd.DataFrame,
    *,
    smiles_col: str = "standardized_smiles",
    label_col: str = "activity_label",
) -> pd.DataFrame:
    """Return actives belonging to the scaffold-training partition only."""

    required_benchmark = {smiles_col, label_col}
    required_assignment = {smiles_col, "split"}
    if missing := sorted(required_benchmark.difference(benchmark.columns)):
        raise KeyError(f"Benchmark table is missing columns: {missing}")
    if missing := sorted(required_assignment.difference(scaffold_assignment.columns)):
        raise KeyError(f"Scaffold assignment is missing columns: {missing}")

    assignment = scaffold_assignment.loc[:, [smiles_col, "split"]].copy()
    if assignment[smiles_col].duplicated().any():
        raise ValueError("Scaffold assignment contains duplicate standardized SMILES.")
    merged = benchmark.merge(assignment, on=smiles_col, how="left", validate="one_to_one")
    if merged["split"].isna().any():
        raise ValueError("Scaffold assignment does not cover every processed EGFR compound.")
    actives = merged.loc[
        merged["split"].eq("train") & merged[label_col].astype(int).eq(1)
    ].copy()
    if actives.empty:
        raise ValueError("Scaffold training split contains no actives.")
    return actives.reset_index(drop=True)


def annotate_nearest_active(
    frame: pd.DataFrame,
    index: NearestActiveIndex,
    *,
    smiles_col: str = "standardized_smiles",
) -> pd.DataFrame:
    """Add nearest training-active SMILES and Tanimoto for each input row."""

    if smiles_col not in frame.columns:
        raise KeyError(f"Missing query SMILES column: {smiles_col!r}")
    out = frame.copy()
    nearest_smiles: list[str] = []
    nearest_tanimoto: list[float] = []
    for smiles in out[smiles_col].astype(str):
        nearest, similarity = index.nearest(smiles)
        nearest_smiles.append(nearest)
        nearest_tanimoto.append(similarity)
    out["nearest_training_active"] = nearest_smiles
    out["nearest_active_tanimoto"] = nearest_tanimoto
    return out


def risk_note(row: pd.Series) -> str:
    """Create a compact triage note from explicit flags, not a medicinal-chemistry verdict."""

    notes: list[str] = []
    similarity = row.get("nearest_active_tanimoto", np.nan)
    if pd.notna(similarity):
        similarity = float(similarity)
        if similarity >= 0.70:
            notes.append("close training-active analog")
        elif similarity >= 0.50:
            notes.append("moderate training-active similarity")
        else:
            notes.append("lower similarity; higher extrapolation risk")
    if bool(row.get("pains", False)):
        notes.append("PAINS alert")
    if not bool(row.get("passes_property_filters", True)):
        notes.append("outside simple property window")
    return "; ".join(notes) if notes else "no flagged issue from configured triage"


def annotate_risk_notes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["risk_note"] = out.apply(risk_note, axis=1)
    return out


def select_interesting_ranked(
    ranked_filtered: pd.DataFrame,
    similarity_index: NearestActiveIndex,
    *,
    score_col: str = "active_probability",
    max_tanimoto: float = 0.50,
    limit: int = 20,
    smiles_col: str = "standardized_smiles",
) -> tuple[pd.DataFrame, int]:
    """Return top-scoring filtered molecules below a novelty-similarity threshold.

    Rows are traversed in score order and similarity evaluation stops after
    ``limit`` qualifying molecules. If fewer qualify, the full filtered library
    is examined. The returned integer records how many rows were examined.
    """

    if limit <= 0:
        raise ValueError("limit must be positive.")
    if not 0.0 <= float(max_tanimoto) <= 1.0:
        raise ValueError("max_tanimoto must be in [0, 1].")
    required = {score_col, smiles_col, "passes_property_filters"}
    if missing := sorted(required.difference(ranked_filtered.columns)):
        raise KeyError(f"Ranked screening frame is missing columns: {missing}")

    candidates = ranked_filtered.loc[ranked_filtered["passes_property_filters"]].copy()
    candidates = candidates.sort_values(
        [score_col, smiles_col], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)

    selected: list[pd.DataFrame] = []
    examined = 0
    for position in range(len(candidates)):
        row_frame = candidates.iloc[[position]].copy()
        row_frame = annotate_nearest_active(row_frame, similarity_index, smiles_col=smiles_col)
        examined += 1
        if float(row_frame.iloc[0]["nearest_active_tanimoto"]) < float(max_tanimoto):
            selected.append(row_frame)
            if len(selected) >= int(limit):
                break

    if not selected:
        empty = candidates.iloc[0:0].copy()
        empty["nearest_training_active"] = pd.Series(dtype="object")
        empty["nearest_active_tanimoto"] = pd.Series(dtype="float64")
        empty["risk_note"] = pd.Series(dtype="object")
        return empty, examined

    output = pd.concat(selected, ignore_index=True)
    output = annotate_risk_notes(output)
    output.insert(0, "interesting_rank", np.arange(1, len(output) + 1, dtype=int))
    return output, examined


def selected_model_name(selection_table: pd.DataFrame) -> str:
    """Read exactly one validation-selected model from the audit table."""

    required = {"model", "selected_for_screen"}
    if missing := sorted(required.difference(selection_table.columns)):
        raise KeyError(f"Primary-model selection table is missing columns: {missing}")

    raw = selection_table["selected_for_screen"]
    if raw.dtype == bool:
        mask = raw
    else:
        mask = raw.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    selected = selection_table.loc[mask, "model"].astype(str).tolist()
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one model selected for screening; found {selected}.")
    model = selected[0]
    if model not in {"RandomForestClassifier", "TorchFingerprintMLP"}:
        raise ValueError(f"Unsupported selected screening model: {model}")
    return model
