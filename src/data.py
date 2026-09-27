"""Chemical cleaning, activity labeling, and simple molecular descriptors.

Cleaning order
--------------
1. Parse SMILES with RDKit.
2. Keep the parent / largest organic fragment using RDKit MolStandardize.
3. Apply simple charge neutralization with ``Uncharger``.
4. Write canonical isomeric SMILES.
5. Drop empty structures and log counts at every step.

Activity rules
--------------
For activities reported in nM, ``pActivity = 9 - log10(nM)``.

Classification defaults:
- active: pActivity >= 6.0 (<= 1,000 nM)
- inactive: pActivity <= 5.0 (>= 10,000 nM)
- 5.0 < pActivity < 6.0: gray zone, excluded

Censored measurements are handled as bounds, not exact values. ``<``/``<=``
measurements are active only when the bound itself guarantees activity;
``>``/``>=`` measurements are inactive only when the bound itself guarantees
inactivity. Other censored measurements remain unlabeled.

Duplicate compounds are grouped by standardized parent SMILES. If both active
and inactive labels occur, the compound is marked conflicting and dropped from
classification. Median pActivity uses exact (``=``) measurements only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

LOGGER = logging.getLogger(__name__)

RELATION_EXACT = "="
RELATION_LOWER_NM = {"<", "<="}   # true nM value is lower; true pActivity is higher
RELATION_UPPER_NM = {">", ">="}   # true nM value is higher; true pActivity is lower
ALLOWED_RELATIONS = {RELATION_EXACT, *RELATION_LOWER_NM, *RELATION_UPPER_NM}


@dataclass
class CleaningReport:
    input_rows: int = 0
    parsed_smiles: int = 0
    parent_fragment: int = 0
    neutralized: int = 0
    canonical_nonempty: int = 0
    output_rows: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class StandardizedMolecule:
    standardized_smiles: str
    scaffold_smiles: str
    mw: float
    clogp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    lipinski_ok: bool
    veber_ok: bool
    qed: float
    pains: bool
    pains_description: str | None
    sa_score: float | None


def pactivity_from_nm(value_nm: float) -> float:
    """Convert a positive nM value to pActivity."""

    value = float(value_nm)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Activity must be a finite positive nM value; got {value_nm!r}.")
    return 9.0 - math.log10(value)


def classify_activity(
    value_nm: float,
    relation: str,
    active_pactivity: float = 6.0,
    inactive_pactivity: float = 5.0,
) -> int | None:
    """Return 1/0 when an activity relation guarantees a class, else ``None``.

    Examples with default thresholds:
    - ``= 500 nM`` -> active
    - ``= 3000 nM`` -> gray zone -> None
    - ``< 1000 nM`` -> active (the true value is even more potent)
    - ``< 5000 nM`` -> ambiguous -> None
    - ``> 10000 nM`` -> inactive (the true value is even less potent)
    - ``> 2000 nM`` -> ambiguous -> None
    """

    relation = str(relation).strip()
    if relation not in ALLOWED_RELATIONS:
        return None

    p_limit = pactivity_from_nm(value_nm)
    if relation == RELATION_EXACT:
        if p_limit >= active_pactivity:
            return 1
        if p_limit <= inactive_pactivity:
            return 0
        return None

    if relation in RELATION_LOWER_NM:
        return 1 if p_limit >= active_pactivity else None

    if relation in RELATION_UPPER_NM:
        return 0 if p_limit <= inactive_pactivity else None

    return None


def add_activity_columns(
    frame: pd.DataFrame,
    value_col: str = "standard_value",
    relation_col: str = "standard_relation",
    active_pactivity: float = 6.0,
    inactive_pactivity: float = 5.0,
) -> pd.DataFrame:
    """Add transformed activity-limit and relation-aware classification columns."""

    out = frame.copy()
    numeric = pd.to_numeric(out[value_col], errors="coerce")
    valid = numeric.gt(0) & np.isfinite(numeric)
    out["pactivity_limit"] = np.nan
    out.loc[valid, "pactivity_limit"] = 9.0 - np.log10(numeric.loc[valid].astype(float))
    out["is_censored"] = out[relation_col].astype(str).ne(RELATION_EXACT)

    labels: list[int | None] = []
    for value, relation in zip(numeric, out[relation_col], strict=False):
        if pd.isna(value) or float(value) <= 0:
            labels.append(None)
            continue
        labels.append(
            classify_activity(
                float(value),
                str(relation),
                active_pactivity=active_pactivity,
                inactive_pactivity=inactive_pactivity,
            )
        )
    out["activity_label"] = pd.array(labels, dtype="Int64")

    # Only exact measurements are honest point estimates of pActivity.
    out["pactivity_exact"] = out["pactivity_limit"].where(~out["is_censored"])
    return out


def _parse_smiles(smiles: Any) -> Chem.Mol | None:
    if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    try:
        return Chem.MolFromSmiles(text)
    except Exception:
        return None


def parent_fragment(mol: Chem.Mol) -> Chem.Mol | None:
    """Return RDKit's standardized parent fragment, or ``None`` on failure."""

    try:
        cleaned = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(cleaned)
        return parent if parent is not None and parent.GetNumAtoms() > 0 else None
    except Exception:
        return None


def neutralize_mol(mol: Chem.Mol) -> Chem.Mol | None:
    """Apply simple RDKit charge neutralization."""

    try:
        result = rdMolStandardize.Uncharger().uncharge(mol)
        return result if result is not None and result.GetNumAtoms() > 0 else None
    except Exception:
        return None


def canonical_isomeric_smiles(mol: Chem.Mol) -> str | None:
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None
    smiles = smiles.strip()
    return smiles or None


def murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    """Return non-chiral Bemis-Murcko scaffold SMILES; empty for acyclic molecules."""

    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def _build_pains_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog.FilterCatalog(params)


_PAINS_CATALOG = _build_pains_catalog()


def pains_flag(mol: Chem.Mol) -> tuple[bool, str | None]:
    """Return the first RDKit PAINS match, if any."""

    match = _PAINS_CATALOG.GetFirstMatch(mol)
    if match is None:
        return False, None
    return True, str(match.GetDescription())


def synthetic_accessibility_score(mol: Chem.Mol) -> float | None:
    """Return RDKit's optional SA score when its Contrib helper is importable."""

    try:
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore
    except Exception:
        return None
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def compute_descriptors(mol: Chem.Mol) -> dict[str, Any]:
    """Compute the small descriptor/filter set used throughout the project."""

    mw = float(Descriptors.MolWt(mol))
    clogp = float(Crippen.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rotatable = int(Lipinski.NumRotatableBonds(mol))
    qed = float(QED.qed(mol))
    pains, pains_description = pains_flag(mol)
    sa = synthetic_accessibility_score(mol)

    # Simple rule flags. They are diagnostics, not medicinal-chemistry verdicts.
    lipinski_ok = mw <= 500.0 and clogp <= 5.0 and hbd <= 5 and hba <= 10
    veber_ok = rotatable <= 10 and tpsa <= 140.0

    return {
        "mw": mw,
        "clogp": clogp,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rotatable,
        "lipinski_ok": lipinski_ok,
        "veber_ok": veber_ok,
        "qed": qed,
        "pains": pains,
        "pains_description": pains_description,
        "sa_score": sa,
        "sa_score_available": sa is not None,
    }


def standardize_molecule(smiles: str) -> StandardizedMolecule | None:
    """Standardize one SMILES and compute scaffold/descriptors."""

    mol = _parse_smiles(smiles)
    if mol is None:
        return None
    mol = parent_fragment(mol)
    if mol is None:
        return None
    mol = neutralize_mol(mol)
    if mol is None:
        return None
    standardized = canonical_isomeric_smiles(mol)
    if standardized is None:
        return None

    # Reparse the canonical string so every downstream calculation uses the
    # exact representation that is persisted to disk.
    canonical_mol = Chem.MolFromSmiles(standardized)
    if canonical_mol is None:
        return None
    desc = compute_descriptors(canonical_mol)
    return StandardizedMolecule(
        standardized_smiles=standardized,
        scaffold_smiles=murcko_scaffold_smiles(canonical_mol),
        mw=desc["mw"],
        clogp=desc["clogp"],
        tpsa=desc["tpsa"],
        hbd=desc["hbd"],
        hba=desc["hba"],
        rotatable_bonds=desc["rotatable_bonds"],
        lipinski_ok=desc["lipinski_ok"],
        veber_ok=desc["veber_ok"],
        qed=desc["qed"],
        pains=desc["pains"],
        pains_description=desc["pains_description"],
        sa_score=desc["sa_score"],
    )


def clean_smiles_dataframe(
    frame: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    log: logging.Logger | None = LOGGER,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Standardize a dataframe while recording counts after every cleaning step."""

    if smiles_col not in frame.columns:
        raise KeyError(f"Missing SMILES column: {smiles_col!r}")

    report = CleaningReport(input_rows=len(frame))
    records: list[dict[str, Any]] = []

    for row_index, smiles in frame[smiles_col].items():
        mol = _parse_smiles(smiles)
        if mol is None:
            continue
        report.parsed_smiles += 1

        mol = parent_fragment(mol)
        if mol is None:
            continue
        report.parent_fragment += 1

        mol = neutralize_mol(mol)
        if mol is None:
            continue
        report.neutralized += 1

        standardized = canonical_isomeric_smiles(mol)
        if standardized is None:
            continue
        report.canonical_nonempty += 1

        canonical_mol = Chem.MolFromSmiles(standardized)
        if canonical_mol is None:
            continue
        desc = compute_descriptors(canonical_mol)
        records.append(
            {
                "_row_index": row_index,
                "standardized_smiles": standardized,
                "scaffold_smiles": murcko_scaffold_smiles(canonical_mol),
                **desc,
            }
        )

    report.output_rows = len(records)
    additions = pd.DataFrame.from_records(records)
    if additions.empty:
        cleaned = frame.iloc[0:0].copy()
        for col in [
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
        ]:
            cleaned[col] = pd.Series(dtype="object")
    else:
        cleaned = frame.loc[additions["_row_index"].tolist()].copy()
        cleaned.index = additions.index
        cleaned = pd.concat(
            [cleaned.reset_index(drop=True), additions.drop(columns=["_row_index"]).reset_index(drop=True)],
            axis=1,
        )

    if log is not None:
        for name, value in report.to_dict().items():
            log.info("cleaning_count %s=%d", name, value)
    return cleaned, report


def aggregate_classification_compounds(
    frame: pd.DataFrame,
    smiles_col: str = "standardized_smiles",
    label_col: str = "activity_label",
    pactivity_exact_col: str = "pactivity_exact",
    type_col: str = "standard_type",
    relation_col: str = "standard_relation",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse activity rows to one classification record per standardized parent.

    Returns:
        (retained_compounds, conflicting_compounds)
    """

    required = {smiles_col, label_col, pactivity_exact_col, relation_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    labeled = frame[frame[label_col].notna()].copy()
    retained: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for smiles, group in labeled.groupby(smiles_col, sort=False, dropna=False):
        labels = sorted({int(v) for v in group[label_col].dropna().tolist()})
        summary = {
            smiles_col: smiles,
            "n_labeled_measurements": int(len(group)),
            "n_exact_measurements": int(group[relation_col].eq(RELATION_EXACT).sum()),
            "n_censored_measurements": int(group[relation_col].ne(RELATION_EXACT).sum()),
            "pactivity_median_exact": float(group[pactivity_exact_col].median())
            if group[pactivity_exact_col].notna().any()
            else np.nan,
        }
        if type_col in group.columns:
            summary["standard_types"] = ";".join(sorted(set(group[type_col].dropna().astype(str))))

        if len(labels) > 1:
            summary["observed_labels"] = ";".join(map(str, labels))
            conflicts.append(summary)
            continue

        summary["activity_label"] = labels[0]
        retained.append(summary)

    retained_df = pd.DataFrame.from_records(retained)
    conflicts_df = pd.DataFrame.from_records(conflicts)
    return retained_df, conflicts_df
