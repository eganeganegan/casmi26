from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from casmi.chemistry import smiles_to_inchikey14

SplitStrategy = Literal["structure", "scaffold", "library"]


def bemis_murcko_scaffold(smiles: str, structure_key: str | None = None) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    value = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
    # Acyclic molecules all have an empty Murcko scaffold; keep their structures separate.
    return value or f"ACYCLIC:{structure_key or smiles_to_inchikey14(smiles)}"


def _balanced_group_folds(groups: pd.Series, n_splits: int, seed: int) -> dict[str, int]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    sizes = groups.value_counts().to_dict()
    rng = np.random.default_rng(seed)
    shuffled = list(sizes)
    rng.shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda group: sizes[group], reverse=True)
    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    assignment: dict[str, int] = {}
    for group in ordered:
        fold = int(np.argmin(fold_sizes))
        assignment[str(group)] = fold
        fold_sizes[fold] += sizes[group]
    return assignment


def create_grouped_splits(
    frame: pd.DataFrame,
    *,
    molecule_col: str = "molecule_id",
    smiles_col: str = "normalized_smiles",
    inchikey_col: str = "inchikey14",
    n_splits: int = 5,
    strategy: SplitStrategy = "structure",
    seed: int = 42,
) -> pd.DataFrame:
    """Return one reproducible, structure-disjoint fold row per molecule."""
    required = {molecule_col, smiles_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    columns = [molecule_col, smiles_col] + ([inchikey_col] if inchikey_col in frame else [])
    molecules = frame[columns].drop_duplicates().copy()
    if inchikey_col not in molecules:
        molecules[inchikey_col] = molecules[smiles_col].map(smiles_to_inchikey14)
    counts = molecules.groupby(molecule_col)[inchikey_col].nunique()
    if (counts > 1).any():
        bad = counts[counts > 1].index[:5].tolist()
        raise ValueError(f"Molecule IDs map to multiple InChIKey14 structures: {bad}")
    molecules = molecules.drop_duplicates(molecule_col)
    molecules["scaffold"] = [
        bemis_murcko_scaffold(smiles, str(key))
        for smiles, key in zip(molecules[smiles_col], molecules[inchikey_col], strict=True)
    ]
    if strategy in {"structure", "library"}:
        molecules["split_group"] = molecules[inchikey_col]
    elif strategy == "scaffold":
        molecules["split_group"] = molecules["scaffold"]
    else:
        raise ValueError(f"Unknown or not-yet-implemented split strategy: {strategy}")
    assignment = _balanced_group_folds(molecules["split_group"].astype(str), n_splits, seed)
    molecules["fold"] = molecules["split_group"].astype(str).map(assignment).astype("int16")
    molecules["cluster_id"] = None
    output = molecules.rename(columns={molecule_col: "molecule_id", inchikey_col: "inchikey14"})
    assert_no_structure_leakage(output, fold_col="fold", key_col="inchikey14")
    return output[["molecule_id", "inchikey14", "scaffold", "cluster_id", "fold"]]


def assert_no_structure_leakage(
    splits: pd.DataFrame, *, fold_col: str = "fold", key_col: str = "inchikey14"
) -> None:
    folds_per_key = splits.groupby(key_col, dropna=False)[fold_col].nunique()
    leaking = folds_per_key[folds_per_key > 1]
    if not leaking.empty:
        raise AssertionError(
            f"Structure leakage: {len(leaking)} {key_col} values occur in multiple folds; "
            f"examples={leaking.index[:5].tolist()}"
        )


def assert_train_validation_disjoint(
    train: pd.DataFrame, validation: pd.DataFrame, key_col: str = "inchikey14"
) -> None:
    overlap = set(train[key_col].dropna()) & set(validation[key_col].dropna())
    if overlap:
        raise AssertionError(
            f"Retrieval leakage: {len(overlap)} validation structures remain in training; "
            f"examples={sorted(overlap)[:5]}"
        )
