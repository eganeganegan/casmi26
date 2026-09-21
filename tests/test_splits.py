import pandas as pd
import pytest

from casmi.validation import (
    assert_no_structure_leakage,
    assert_train_validation_disjoint,
    create_grouped_splits,
)


def test_grouped_structure_split_keeps_duplicate_structure_together() -> None:
    frame = pd.DataFrame(
        {
            "molecule_id": ["a", "a", "b", "c", "d", "e"],
            "normalized_smiles": ["CCO", "CCO", "OCC", "CCC", "CCCC", "c1ccccc1"],
        }
    )
    splits = create_grouped_splits(frame, n_splits=3)
    fold_a = splits.loc[splits.molecule_id == "a", "fold"].item()
    fold_b = splits.loc[splits.molecule_id == "b", "fold"].item()
    assert fold_a == fold_b
    assert_no_structure_leakage(splits)


def test_explicit_leakage_assertion() -> None:
    train = pd.DataFrame({"inchikey14": ["AAA", "BBB"]})
    validation = pd.DataFrame({"inchikey14": ["BBB", "CCC"]})
    with pytest.raises(AssertionError, match="Retrieval leakage"):
        assert_train_validation_disjoint(train, validation)


def test_multiple_smiles_for_same_competition_key_are_allowed() -> None:
    frame = pd.DataFrame(
        {
            "molecule_id": ["same", "same", "other"],
            "normalized_smiles": ["F[C@H](Cl)Br", "F[C@@H](Cl)Br", "CCO"],
            "inchikey14": ["YACLCMMBHTUQON", "YACLCMMBHTUQON", "LFQSCWFLJHTTHZ"],
        }
    )
    splits = create_grouped_splits(frame, n_splits=2)
    assert len(splits) == 2
