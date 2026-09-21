from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


@lru_cache(maxsize=500_000)
def _mol_from_smiles(smiles: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    Chem.SanitizeMol(mol)
    return mol


def sanitize_smiles(smiles: str) -> bool:
    """Return whether RDKit can parse and fully sanitize a SMILES string."""
    try:
        _mol_from_smiles(smiles)
        return True
    except (ValueError, RuntimeError):
        return False


@lru_cache(maxsize=500_000)
def canonicalize_smiles(smiles: str, isomeric: bool = True) -> str:
    mol = _mol_from_smiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


@lru_cache(maxsize=500_000)
def tautomer_canonicalize_smiles(smiles: str) -> str:
    """Canonicalize with RDKit's default tautomer enumerator, matching the metric."""
    mol = _mol_from_smiles(smiles)
    canonical = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    return Chem.MolToSmiles(canonical, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=500_000)
def smiles_to_inchikey14(smiles: str) -> str:
    tautomer_smiles = tautomer_canonicalize_smiles(smiles)
    key = Chem.MolToInchiKey(_mol_from_smiles(tautomer_smiles))
    if not key:
        raise ValueError(f"Could not generate InChIKey for {smiles!r}")
    return key.split("-", maxsplit=1)[0]


@lru_cache(maxsize=500_000)
def exact_mass(smiles: str) -> float:
    return float(rdMolDescriptors.CalcExactMolWt(_mol_from_smiles(smiles)))


@lru_cache(maxsize=500_000)
def molecular_formula(smiles: str) -> str:
    return str(rdMolDescriptors.CalcMolFormula(_mol_from_smiles(smiles)))


@lru_cache(maxsize=200_000)
def _morgan_bitvect(smiles: str, radius: int, n_bits: int):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return generator.GetFingerprint(_mol_from_smiles(smiles))


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(_morgan_bitvect(smiles, radius, n_bits), arr)
    return arr


@lru_cache(maxsize=200_000)
def _rdkit_bitvect(smiles: str, n_bits: int):
    generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
    return generator.GetFingerprint(_mol_from_smiles(smiles))


def rdkit_fingerprint(smiles: str, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(_rdkit_bitvect(smiles, n_bits), arr)
    return arr


def tanimoto(fp1: np.ndarray, fp2: np.ndarray) -> float:
    a = np.asarray(fp1, dtype=bool)
    b = np.asarray(fp2, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Fingerprint shapes differ: {a.shape} != {b.shape}")
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def deduplicate_structures_by_inchikey14(
    smiles: Iterable[str], limit: int | None = None
) -> list[str]:
    """Keep the first valid representative of each competition comparison key."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    result: list[str] = []
    seen: set[str] = set()
    if limit == 0:
        return result
    for value in smiles:
        try:
            key = smiles_to_inchikey14(value)
        except (ValueError, RuntimeError):
            continue
        if key not in seen:
            seen.add(key)
            result.append(value)
            if limit is not None and len(result) >= limit:
                break
    return result


@lru_cache(maxsize=500_000)
def basic_descriptors(smiles: str) -> dict[str, float]:
    mol = _mol_from_smiles(smiles)
    heavy = max(1, mol.GetNumHeavyAtoms())
    return {
        "exact_mass": exact_mass(smiles),
        "heteroatom_count": float(Lipinski.NumHeteroatoms(mol)),
        "ring_count": float(Lipinski.RingCount(mol)),
        "aromatic_fraction": float(sum(a.GetIsAromatic() for a in mol.GetAtoms()) / heavy),
        "logp": float(Descriptors.MolLogP(mol)),
    }
