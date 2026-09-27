import pytest

from src.data import classify_activity, pactivity_from_nm


def test_pactivity_cutoffs():
    assert pactivity_from_nm(1000.0) == pytest.approx(6.0)
    assert pactivity_from_nm(10000.0) == pytest.approx(5.0)


def test_exact_activity_labels():
    assert classify_activity(1000.0, "=") == 1
    assert classify_activity(10000.0, "=") == 0
    assert classify_activity(3000.0, "=") is None


def test_censored_labels_only_when_guaranteed():
    assert classify_activity(1000.0, "<") == 1
    assert classify_activity(5000.0, "<") is None
    assert classify_activity(10000.0, ">") == 0
    assert classify_activity(2000.0, ">") is None


def test_duplicate_conflict_is_dropped():
    import pandas as pd

    from src.data import aggregate_classification_compounds

    frame = pd.DataFrame(
        {
            "standardized_smiles": ["CCO", "CCO", "CCN"],
            "activity_label": pd.array([1, 0, 1], dtype="Int64"),
            "pactivity_exact": [7.0, 4.0, 6.5],
            "standard_relation": ["=", "=", "="],
            "standard_type": ["IC50", "Ki", "IC50"],
        }
    )
    retained, conflicts = aggregate_classification_compounds(frame)
    assert retained["standardized_smiles"].tolist() == ["CCN"]
    assert conflicts["standardized_smiles"].tolist() == ["CCO"]
