import numpy as np
import pandas as pd

from src.features import DESCRIPTOR_COLUMNS, feature_matrix_from_frame, morgan_fingerprint


def test_morgan_fingerprint_shape_and_binary_values():
    fp = morgan_fingerprint("CCO", radius=2, n_bits=128)
    assert fp.shape == (128,)
    assert set(np.unique(fp)).issubset({0.0, 1.0})


def test_feature_matrix_concatenates_descriptors():
    row = {
        "standardized_smiles": "CCO",
        "mw": 46.07,
        "clogp": -0.1,
        "tpsa": 20.2,
        "hbd": 1,
        "hba": 1,
        "rotatable_bonds": 0,
        "qed": 0.4,
    }
    X = feature_matrix_from_frame(pd.DataFrame([row]), n_bits=64)
    assert X.shape == (1, 64 + len(DESCRIPTOR_COLUMNS))
    assert np.isfinite(X).all()
