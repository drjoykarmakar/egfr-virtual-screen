from pathlib import Path

import pandas as pd
import pytest

from src.screen import (
    NearestActiveIndex,
    annotate_nearest_active,
    apply_screen_filters,
    prepare_screening_library,
    read_screening_library,
    select_interesting_ranked,
    selected_model_name,
    training_actives_from_assignment,
)


def test_read_smi_library_uses_first_token_and_optional_id(tmp_path: Path):
    path = tmp_path / "library.smi"
    path.write_text("# comment\nCCO cmp1\nCCN\n\n", encoding="utf-8")
    frame = read_screening_library(path)
    assert frame["input_smiles"].tolist() == ["CCO", "CCN"]
    assert frame["library_id"].tolist() == ["cmp1", "row_3"]


def test_prepare_library_standardizes_deduplicates_and_removes_benchmark_overlap():
    raw = pd.DataFrame(
        {
            "library_id": ["a", "b", "c", "bad"],
            "input_smiles": ["CCO", "OCC", "CCN", "not-a-smiles"],
        }
    )
    prepared, report = prepare_screening_library(raw, benchmark_smiles=["CCN"])
    assert prepared["standardized_smiles"].tolist() == ["CCO"]
    assert report.raw_rows == 4
    assert report.standardized_rows == 3
    assert report.standardized_duplicates_removed == 1
    assert report.exact_benchmark_overlaps_removed == 1
    assert report.final_rows == 1


def test_filters_are_inclusive_and_pains_does_not_delete_or_fail_row():
    frame = pd.DataFrame(
        {
            "mw": [200.0, 600.0, 199.9, 300.0],
            "clogp": [-1.0, 5.0, 1.0, 5.1],
            "qed": [0.40, 0.90, 0.80, 0.80],
            "pains": [True, False, False, False],
        }
    )
    filtered = apply_screen_filters(frame)
    assert filtered["passes_property_filters"].tolist() == [True, True, False, False]
    assert bool(filtered.loc[0, "pains"]) is True
    assert bool(filtered.loc[0, "passes_property_filters"]) is True


def test_training_active_reference_uses_scaffold_train_only():
    benchmark = pd.DataFrame(
        {
            "standardized_smiles": ["CCO", "CCN", "CCC", "CCCl"],
            "activity_label": [1, 1, 0, 1],
        }
    )
    assignment = pd.DataFrame(
        {
            "standardized_smiles": ["CCO", "CCN", "CCC", "CCCl"],
            "split": ["train", "validation", "train", "test"],
        }
    )
    actives = training_actives_from_assignment(benchmark, assignment)
    assert actives["standardized_smiles"].tolist() == ["CCO"]


def test_nearest_active_returns_identity_for_identical_smiles():
    index = NearestActiveIndex.from_smiles(["CCO", "c1ccccc1"])
    queries = pd.DataFrame({"standardized_smiles": ["CCO", "CCN"]})
    annotated = annotate_nearest_active(queries, index)
    assert annotated.loc[0, "nearest_training_active"] == "CCO"
    assert annotated.loc[0, "nearest_active_tanimoto"] == pytest.approx(1.0)
    assert annotated["nearest_active_tanimoto"].between(0.0, 1.0).all()


def test_interesting_selection_is_filtered_score_order_and_strictly_below_threshold():
    index = NearestActiveIndex.from_smiles(["CCO"])
    ranked = pd.DataFrame(
        {
            "standardized_smiles": ["CCO", "c1ccccc1", "C1CCCCC1", "CCN"],
            "active_probability": [0.99, 0.90, 0.80, 0.70],
            "passes_property_filters": [True, False, True, True],
            "pains": [False, False, False, False],
        }
    )
    selected, examined = select_interesting_ranked(
        ranked, index, max_tanimoto=0.5, limit=1
    )
    assert len(selected) == 1
    # benzene is skipped before similarity because it fails property filters;
    # C1CCCCC1 is therefore the first qualifying filtered low-similarity row.
    assert selected.iloc[0]["standardized_smiles"] == "C1CCCCC1"
    assert float(selected.iloc[0]["nearest_active_tanimoto"]) < 0.5
    assert examined == 2


def test_selected_model_name_requires_exactly_one_supported_selection():
    table = pd.DataFrame(
        {
            "model": ["RandomForestClassifier", "TorchFingerprintMLP"],
            "selected_for_screen": [False, True],
        }
    )
    assert selected_model_name(table) == "TorchFingerprintMLP"

    both = table.copy()
    both["selected_for_screen"] = True
    with pytest.raises(ValueError, match="exactly one"):
        selected_model_name(both)
