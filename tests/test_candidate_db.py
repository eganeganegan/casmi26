import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from casmi.retrieval import (
    CandidateDatabase,
    build_candidate_database,
    build_numpy_candidate_database,
    merge_candidate_databases,
    normalize_candidate_isotopes,
)


def test_build_and_query_candidate_database(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "candidates.parquet"
    pd.DataFrame(
        {
            "identifier": ["a", "a-duplicate", "b", "invalid"],
            "canonical_smiles": ["CCO", "OCC", "CCC", None],
            "standard_inchi_key": [
                "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "ATUOYWHBWRKTHZ-UHFFFAOYSA-N",
                None,
            ],
            "exact_molecular_weight": [46.041865, 46.041865, 44.062600, None],
            "molecular_formula": ["C2H6O", "C2H6O", "C3H8", None],
        }
    ).to_csv(source, index=False)
    report = build_candidate_database(
        source,
        output,
        smiles_column="canonical_smiles",
        exact_mass_column="exact_molecular_weight",
        source="test",
        candidate_id_column="identifier",
        inchikey_column="standard_inchi_key",
        formula_column="molecular_formula",
    )
    assert report == {"input_rows": 4, "output_rows": 2, "duplicates_or_invalid": 2}

    database = CandidateDatabase.from_parquet(output)
    assert len(database) == 2
    mass_hits = database.query_mass(46.041865, ppm=5)
    assert [hit.inchikey14 for hit in mass_hits] == ["LFQSCWFLJHTTHZ"]
    assert mass_hits[0].mass_error_ppm == pytest.approx(0)
    assert database.query_formula("C3H8")[0].smiles == "CCC"
    assert database.query_mass_and_formula(46.041865, 5, "C2H6O")[0].smiles in {"CCO", "OCC"}


def test_mass_channels_preserve_narrow_and_fallback_membership(tmp_path: Path) -> None:
    database = CandidateDatabase(
        pl.DataFrame(
            {
                "candidate_id": ["near", "fallback"],
                "smiles": ["CCO", "CCC"],
                "canonical_smiles": ["CCO", "CCC"],
                "inchikey14": ["LFQSCWFLJHTTHZ", "ATUOYWHBWRKTHZ"],
                "formula": ["C2H6O", "C3H8"],
                "exact_mass": [100.001, 100.050],
                "source": ["test", "test"],
            }
        )
    )
    hits = database.query_mass_channels(100.0, [20, 1000, 100])
    assert [hit.candidate.candidate_id for hit in hits] == ["near", "fallback"]
    assert hits[0].min_tolerance_ppm == 20
    assert hits[0].channels_ppm == (20.0, 100.0, 1000.0)
    assert hits[1].min_tolerance_ppm == 1000
    assert hits[1].channels_ppm == (1000.0,)


def test_mass_channels_validate_tolerances() -> None:
    database = CandidateDatabase(
        pl.DataFrame(
            {
                "candidate_id": ["a"],
                "smiles": ["C"],
                "canonical_smiles": ["C"],
                "inchikey14": ["VNWKTOKETHGBQD"],
                "formula": ["CH4"],
                "exact_mass": [16.0313],
                "source": ["test"],
            }
        )
    )
    with pytest.raises(ValueError, match="At least one"):
        database.query_mass_channels(16.0, [])
    with pytest.raises(ValueError, match="finite and non-negative"):
        database.query_mass_channels(16.0, [float("nan")])


def test_safe_numpy_bundle_and_merge(tmp_path: Path) -> None:
    metadata = tmp_path / "meta.pkl"
    masses = tmp_path / "mass.npy"
    bio_output = tmp_path / "bio.parquet"
    merged_output = tmp_path / "merged.parquet"
    with metadata.open("wb") as handle:
        pickle.dump(
            {
                "keys": np.asarray(["LFQSCWFLJHTTHZ", "ATUOYWHBWRKTHZ"], dtype=object),
                "smiles": np.asarray(["CCO", "CCC"], dtype=object),
            },
            handle,
        )
    np.save(masses, np.asarray([46.041865, 44.062600]))
    report = build_numpy_candidate_database(
        metadata, masses, bio_output, source="bio"
    )
    assert report["output_rows"] == 2
    merged = merge_candidate_databases([bio_output, bio_output], merged_output)
    assert merged == {"input_rows": 4, "output_rows": 2, "duplicates_removed": 2}


def test_restricted_pickle_blocks_globals(tmp_path: Path) -> None:
    metadata = tmp_path / "unsafe.pkl"
    masses = tmp_path / "mass.npy"
    metadata.write_bytes(pickle.dumps(Path("unsafe")))
    np.save(masses, np.asarray([1.0]))
    with pytest.raises(pickle.UnpicklingError, match="Blocked unsafe pickle global"):
        build_numpy_candidate_database(metadata, masses, tmp_path / "out.parquet", source="bad")


def test_isotope_normalization_repairs_candidate_mass(tmp_path: Path) -> None:
    source = tmp_path / "isotope.parquet"
    output = tmp_path / "normal.parquet"
    CandidateDatabase(
        pl.DataFrame(
            {
                "candidate_id": ["isotope"],
                "smiles": ["[11CH3]CO"],
                "canonical_smiles": ["[11CH3]CO"],
                "inchikey14": ["LFQSCWFLJHTTHZ"],
                "formula": ["C2H6O"],
                "exact_mass": [45.053045],
                "source": ["test"],
            }
        )
    ).save(source)
    report = normalize_candidate_isotopes(source, output)
    assert report["isotope_rows_normalized"] == 1
    record = CandidateDatabase.from_parquet(output).query_mass(46.041865, 5)[0]
    assert record.smiles == "CCO"
    assert record.exact_mass == pytest.approx(46.041865, abs=1e-6)
