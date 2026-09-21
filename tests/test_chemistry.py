import pytest

from casmi.chemistry import (
    canonicalize_smiles,
    deduplicate_structures_by_inchikey14,
    mass_error_ppm,
    morgan_fingerprint,
    sanitize_smiles,
    smiles_to_inchikey14,
    tanimoto,
    tautomer_canonicalize_smiles,
)


def test_canonicalization_and_sanitization() -> None:
    assert canonicalize_smiles("OCC") == "CCO"
    assert sanitize_smiles("CCO")
    assert not sanitize_smiles("not a smiles")


def test_stereoisomers_share_connectivity_key() -> None:
    assert smiles_to_inchikey14("F[C@H](Cl)Br") == smiles_to_inchikey14("F[C@@H](Cl)Br")


def test_tautomers_share_metric_key() -> None:
    keto = "CC(=O)CC(C)=O"
    enol = "CC(=O)C=C(C)O"
    assert tautomer_canonicalize_smiles(keto) == tautomer_canonicalize_smiles(enol)
    assert smiles_to_inchikey14(keto) == smiles_to_inchikey14(enol)


def test_fingerprint_and_tanimoto() -> None:
    fp = morgan_fingerprint("CCO")
    assert fp.shape == (2048,)
    assert tanimoto(fp, fp) == 1.0
    assert 0 <= tanimoto(fp, morgan_fingerprint("c1ccccc1")) < 1


def test_deduplicate_structure_connectivity() -> None:
    values = ["F[C@H](Cl)Br", "F[C@@H](Cl)Br", "CCO", "invalid"]
    assert deduplicate_structures_by_inchikey14(values) == ["F[C@H](Cl)Br", "CCO"]
    assert deduplicate_structures_by_inchikey14(values, limit=1) == ["F[C@H](Cl)Br"]
    assert deduplicate_structures_by_inchikey14(values, limit=0) == []
    with pytest.raises(ValueError, match="non-negative"):
        deduplicate_structures_by_inchikey14(values, limit=-1)


def test_mass_error_ppm() -> None:
    assert mass_error_ppm(100.001, 100.0) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        mass_error_ppm(1, 0)
