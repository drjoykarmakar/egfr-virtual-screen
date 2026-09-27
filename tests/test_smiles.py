from rdkit import Chem

from src.data import standardize_molecule


def test_salt_is_reduced_to_parent_fragment():
    result = standardize_molecule("CC(=O)[O-].[Na+]")
    assert result is not None
    mol = Chem.MolFromSmiles(result.standardized_smiles)
    assert mol is not None
    assert len(Chem.GetMolFrags(mol)) == 1


def test_invalid_smiles_returns_none():
    assert standardize_molecule("not_a_smiles") is None
