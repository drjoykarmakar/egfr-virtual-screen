"""Small plotting helpers for the benchmark report."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def plot_nearest_active_similarity(
    similarities: Iterable[float],
    output_path: str | Path,
    *,
    bins: int = 20,
) -> None:
    """Save the key top-score nearest-training-active Tanimoto histogram."""

    values = np.asarray(list(similarities), dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite similarities were supplied for the histogram.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(values, bins=int(bins), range=(0.0, 1.0), edgecolor="black")
    ax.axvline(0.5, linestyle="--", linewidth=1.2, label="interesting threshold = 0.5")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Tanimoto to nearest scaffold-training active")
    ax.set_ylabel("Top-ranked molecules")
    ax.set_title("Nearest-training-active similarity among top model scores")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_molecule_grid(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    smiles_col: str = "standardized_smiles",
    score_col: str = "active_probability",
    similarity_col: str = "nearest_active_tanimoto",
    max_molecules: int = 20,
) -> None:
    """Save a compact RDKit grid for the low-similarity shortlist."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No molecules met the configured filtered + low-similarity shortlist criteria.",
            ha="center",
            va="center",
            wrap=True,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=180)
        plt.close(fig)
        return
    required = {smiles_col, score_col, similarity_col}
    if missing := sorted(required.difference(frame.columns)):
        raise KeyError(f"Molecule-grid frame is missing columns: {missing}")

    subset = frame.head(int(max_molecules))
    molecules = [Chem.MolFromSmiles(str(value)) for value in subset[smiles_col]]
    if any(mol is None for mol in molecules):
        raise ValueError("Molecule-grid input contains invalid SMILES.")
    legends = [
        f"#{idx + 1} score={float(score):.3f}\nTanimoto={float(similarity):.2f}"
        for idx, (score, similarity) in enumerate(
            zip(subset[score_col], subset[similarity_col], strict=False)
        )
    ]
    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=4,
        subImgSize=(260, 210),
        legends=legends,
        useSVG=False,
    )
    image.save(str(output))
