"""Random and Bemis-Murcko scaffold splits.

The scaffold splitter is deliberately label-blind. Whole scaffold groups are
assigned to one partition using a deterministic greedy size objective, so no
Bemis-Murcko scaffold can appear in more than one split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def _validate_fractions(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> np.ndarray:
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction], dtype=float
    )
    if np.any(fractions <= 0):
        raise ValueError("All split fractions must be positive.")
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError(f"Split fractions must sum to 1.0; got {fractions.sum():.6f}.")
    return fractions


def random_split_indices(
    labels: Iterable[int],
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 2026,
    stratify: bool = True,
) -> SplitIndices:
    """Return a reproducible random split, stratifying labels when possible."""

    _validate_fractions(train_fraction, validation_fraction, test_fraction)
    y = np.asarray(list(labels))
    n = len(y)
    if n < 3:
        raise ValueError("At least three rows are required for train/validation/test splits.")

    indices = np.arange(n)
    stratify_values = y if stratify and _can_stratify(y) else None
    train_idx, remainder_idx = train_test_split(
        indices,
        train_size=train_fraction,
        random_state=int(seed),
        shuffle=True,
        stratify=stratify_values,
    )

    remainder_fraction = validation_fraction + test_fraction
    relative_validation = validation_fraction / remainder_fraction
    remainder_y = y[remainder_idx]
    remainder_stratify = remainder_y if stratify and _can_stratify(remainder_y) else None
    validation_idx, test_idx = train_test_split(
        remainder_idx,
        train_size=relative_validation,
        random_state=int(seed) + 1,
        shuffle=True,
        stratify=remainder_stratify,
    )
    return SplitIndices(
        train=np.sort(train_idx),
        validation=np.sort(validation_idx),
        test=np.sort(test_idx),
    )


def _can_stratify(labels: np.ndarray) -> bool:
    if labels.size < 4:
        return False
    _, counts = np.unique(labels, return_counts=True)
    return len(counts) > 1 and bool(np.all(counts >= 2))


def scaffold_split_indices(
    scaffolds: Iterable[str],
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.10,
) -> SplitIndices:
    """Assign whole scaffold groups to splits with zero scaffold overlap.

    Groups are processed largest-first. For each group, the candidate split that
    minimizes normalized squared deviation from the requested target sizes is
    chosen. The procedure does not inspect activity labels.
    """

    fractions = _validate_fractions(train_fraction, validation_fraction, test_fraction)
    scaffold_values = ["" if value is None else str(value) for value in scaffolds]
    n = len(scaffold_values)
    if n < 3:
        raise ValueError("At least three rows are required for train/validation/test splits.")

    groups: dict[str, list[int]] = {}
    for index, scaffold in enumerate(scaffold_values):
        groups.setdefault(scaffold, []).append(index)
    if len(groups) < 3:
        raise ValueError(
            "Scaffold split requires at least three unique Bemis-Murcko scaffolds."
        )

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    targets = fractions * n
    counts = np.zeros(3, dtype=float)
    assigned: list[list[int]] = [[], [], []]

    for _, group_indices in ordered_groups:
        group_size = len(group_indices)
        scores: list[float] = []
        for split_index in range(3):
            candidate = counts.copy()
            candidate[split_index] += group_size
            # Normalize by target size so validation/test are not ignored simply
            # because the training target is much larger.
            score = float(np.sum(((candidate - targets) ** 2) / np.maximum(targets, 1.0)))
            scores.append(score)
        chosen = int(np.argmin(scores))
        assigned[chosen].extend(group_indices)
        counts[chosen] += group_size

    # A pathological scaffold-size distribution can leave a small partition
    # empty. Fail loudly instead of pretending the requested benchmark exists.
    if any(len(partition) == 0 for partition in assigned):
        raise ValueError(
            "Could not construct non-empty scaffold train/validation/test partitions; "
            "inspect scaffold-size distribution."
        )

    split = SplitIndices(
        train=np.asarray(sorted(assigned[0]), dtype=int),
        validation=np.asarray(sorted(assigned[1]), dtype=int),
        test=np.asarray(sorted(assigned[2]), dtype=int),
    )
    assert_zero_scaffold_overlap(scaffold_values, split)
    return split


def split_assignment_frame(
    frame: pd.DataFrame,
    split: SplitIndices,
    smiles_col: str = "standardized_smiles",
    scaffold_col: str = "scaffold_smiles",
    label_col: str = "activity_label",
) -> pd.DataFrame:
    """Return a compact auditable table mapping compounds to split names."""

    result = frame.loc[:, [smiles_col, scaffold_col, label_col]].copy().reset_index(drop=True)
    result["split"] = ""
    for name, indices in split.as_dict().items():
        result.loc[indices, "split"] = name
    if (result["split"] == "").any():
        raise RuntimeError("Some rows were not assigned to a split.")
    return result


def scaffold_overlap(
    scaffolds: Iterable[str],
    split: SplitIndices,
) -> dict[str, set[str]]:
    """Return pairwise scaffold intersections for an assigned split."""

    values = np.asarray(["" if value is None else str(value) for value in scaffolds], dtype=object)
    sets = {
        name: set(values[indices].tolist())
        for name, indices in split.as_dict().items()
    }
    return {
        "train_validation": sets["train"] & sets["validation"],
        "train_test": sets["train"] & sets["test"],
        "validation_test": sets["validation"] & sets["test"],
    }


def assert_zero_scaffold_overlap(
    scaffolds: Iterable[str],
    split: SplitIndices,
) -> None:
    overlaps = scaffold_overlap(scaffolds, split)
    nonempty = {name: values for name, values in overlaps.items() if values}
    if nonempty:
        raise AssertionError(f"Scaffold overlap detected: {nonempty}")


def split_summary(
    frame: pd.DataFrame,
    split: SplitIndices,
    scaffold_col: str = "scaffold_smiles",
    label_col: str = "activity_label",
) -> pd.DataFrame:
    """Summarize row, scaffold, and class counts for each partition."""

    records: list[dict[str, int | str]] = []
    for name, indices in split.as_dict().items():
        part = frame.iloc[indices]
        records.append(
            {
                "split": name,
                "n_compounds": int(len(part)),
                "n_scaffolds": int(part[scaffold_col].nunique(dropna=False)),
                "n_active": int((part[label_col].astype(int) == 1).sum()),
                "n_inactive": int((part[label_col].astype(int) == 0).sum()),
            }
        )
    return pd.DataFrame.from_records(records)
