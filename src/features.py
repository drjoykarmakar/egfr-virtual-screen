"""Molecular features for EGFR activity models.

The project deliberately uses a small, transparent representation:

- Morgan fingerprint, radius 2, 2048 bits by default.
- Seven RDKit descriptors already persisted by :mod:`src.data`.

The same feature builder is used by the Random Forest baseline and the later
PyTorch MLP so model comparisons are not confounded by different inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

DESCRIPTOR_COLUMNS: tuple[str, ...] = (
    "mw",
    "clogp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "qed",
)


def morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> np.ndarray:
    """Return a dense float32 Morgan bit vector for one SMILES.

    Raises:
        ValueError: if the SMILES cannot be parsed.
    """

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Could not parse SMILES for fingerprinting: {smiles!r}")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(n_bits),
        includeChirality=bool(use_chirality),
    )
    return generator.GetFingerprintAsNumPy(mol).astype(np.float32, copy=False)


def morgan_matrix(
    smiles: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> np.ndarray:
    """Fingerprint a sequence of standardized SMILES into a dense matrix."""

    rows = [
        morgan_fingerprint(
            value,
            radius=radius,
            n_bits=n_bits,
            use_chirality=use_chirality,
        )
        for value in smiles
    ]
    if not rows:
        return np.empty((0, int(n_bits)), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)


def descriptor_matrix(
    frame: pd.DataFrame,
    descriptor_columns: Sequence[str] = DESCRIPTOR_COLUMNS,
) -> np.ndarray:
    """Return persisted RDKit descriptors as a float32 matrix.

    Descriptor computation belongs in ``src.data``. This function intentionally
    refuses missing/non-finite values rather than silently imputing them.
    """

    missing = [column for column in descriptor_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing descriptor columns: {missing}")

    values = frame.loc[:, list(descriptor_columns)].to_numpy(dtype=np.float32)
    if values.size and not np.isfinite(values).all():
        raise ValueError("Descriptor matrix contains NaN or infinite values.")
    return values


def feature_matrix_from_frame(
    frame: pd.DataFrame,
    smiles_col: str = "standardized_smiles",
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
    descriptor_columns: Sequence[str] = DESCRIPTOR_COLUMNS,
) -> np.ndarray:
    """Build Morgan + descriptor features from a processed compound table."""

    if smiles_col not in frame.columns:
        raise KeyError(f"Missing SMILES column: {smiles_col!r}")

    fingerprints = morgan_matrix(
        frame[smiles_col].astype(str).tolist(),
        radius=radius,
        n_bits=n_bits,
        use_chirality=use_chirality,
    )
    descriptors = descriptor_matrix(frame, descriptor_columns=descriptor_columns)
    if fingerprints.shape[0] != descriptors.shape[0]:
        raise RuntimeError("Fingerprint and descriptor row counts do not match.")
    return np.hstack([fingerprints, descriptors]).astype(np.float32, copy=False)


def feature_names(
    n_bits: int = 2048,
    descriptor_columns: Sequence[str] = DESCRIPTOR_COLUMNS,
) -> list[str]:
    """Return feature names in the exact matrix column order."""

    return [f"morgan_{index}" for index in range(int(n_bits))] + list(descriptor_columns)
