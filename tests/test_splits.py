import numpy as np
import pandas as pd

from src.splits import (
    assert_zero_scaffold_overlap,
    random_split_indices,
    scaffold_overlap,
    scaffold_split_indices,
    split_summary,
)


def test_scaffold_split_has_zero_overlap_and_complete_assignment():
    scaffolds = [
        "c1ccccc1",
        "c1ccccc1",
        "c1ccncc1",
        "c1ccncc1",
        "C1CCCCC1",
        "C1CCCCC1",
        "c1ncccc1",
        "c1ncccc1",
        "C1CCNCC1",
        "C1CCOCC1",
        "c1ccc2ccccc2c1",
        "c1ccc2[nH]ccc2c1",
        "C1CCC2CCCCC2C1",
        "c1cncnc1",
        "c1ccoc1",
        "c1ccsc1",
        "C1CCNC1",
        "C1CCOC1",
        "N1CCCCC1",
        "O1CCCCC1",
    ]
    split = scaffold_split_indices(scaffolds)
    assert_zero_scaffold_overlap(scaffolds, split)
    overlap = scaffold_overlap(scaffolds, split)
    assert all(not values for values in overlap.values())

    assigned = np.concatenate([split.train, split.validation, split.test])
    assert sorted(assigned.tolist()) == list(range(len(scaffolds)))
    assert all(len(indices) > 0 for indices in split.as_dict().values())


def test_random_split_is_reproducible_and_roughly_stratified():
    labels = np.array([0] * 50 + [1] * 50)
    first = random_split_indices(labels, seed=7)
    second = random_split_indices(labels, seed=7)
    for name in first.as_dict():
        assert np.array_equal(first.as_dict()[name], second.as_dict()[name])

    for indices in first.as_dict().values():
        prevalence = labels[indices].mean()
        assert 0.35 <= prevalence <= 0.65


def test_split_summary_counts_scaffolds_and_labels():
    frame = pd.DataFrame(
        {
            "standardized_smiles": [f"mol{i}" for i in range(12)],
            "scaffold_smiles": [f"scaf{i // 2}" for i in range(12)],
            "activity_label": [0, 1] * 6,
        }
    )
    split = scaffold_split_indices(frame["scaffold_smiles"], 0.5, 0.25, 0.25)
    summary = split_summary(frame, split)
    assert summary["n_compounds"].sum() == len(frame)
    assert set(summary["split"]) == {"train", "validation", "test"}
