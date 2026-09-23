import numpy as np
import pandas as pd

from casmi.chemistry import exact_mass, theoretical_precursor_mz
from casmi.retrieval import (
    CompactSpectralLibraryIndex,
    RawRepresentativeEntropyIndex,
    RepresentativeEntropyIndex,
)
from casmi.spectra import Spectrum, SpectrumPreprocessingConfig


def _spectrum(
    key: str,
    spectrum_id: str,
    smiles: str,
    mass: float,
    mz: list[float],
    intensity: list[float],
) -> Spectrum:
    return Spectrum(
        molecule_id=key,
        spectrum_id=spectrum_id,
        precursor_mz=theoretical_precursor_mz(mass, "[M+H]+"),
        adduct="[M+H]+",
        collision_energy=20.0,
        mz=np.asarray(mz),
        intensity=np.asarray(intensity),
        smiles=smiles,
        inchikey14=key,
        exact_mass=mass,
        ionization_mode="positive",
    )


def test_representative_index_uses_richest_spectrum_and_shifted_match() -> None:
    reference_mass = exact_mass("CCO")
    query_mass = reference_mass + 14.0
    library = CompactSpectralLibraryIndex(
        [
            _spectrum("REFERENCEKEY1", "sparse", "CCO", reference_mass, [10], [1]),
            _spectrum(
                "REFERENCEKEY1", "rich", "CCO", reference_mass, [10, 20, 30], [1, 0.8, 0.4]
            ),
            _spectrum("DISTRACTORKEY", "other", "CCN", query_mass - 5, [8, 17], [1, 1]),
        ],
        preprocessing=SpectrumPreprocessingConfig(remove_precursor_window_da=None),
    )
    index = RepresentativeEntropyIndex(library)
    query = _spectrum(
        "QUERYSTRUCTURE", "query", "CCCO", query_mass, [24, 34, 44], [1, 0.8, 0.4]
    )

    hits = index.search_molecule([query], mass_window_da=30, top_n=2)

    assert len(index) == 2
    assert hits[0].inchikey14 == "REFERENCEKEY1"
    assert hits[0].library_spectrum_index == 1
    assert hits[0].similarity > 0.99
    assert abs(hits[0].mass_delta - 14.0) < 1e-6


def test_representative_index_excludes_query_structure() -> None:
    mass = exact_mass("CCO")
    library = CompactSpectralLibraryIndex(
        [
            _spectrum("SAMESTRUCTURE1", "same", "CCO", mass, [10, 20], [1, 1]),
            _spectrum("OTHERSTRUCTURE", "other", "CCN", mass + 1, [11, 21], [1, 1]),
        ],
        preprocessing=SpectrumPreprocessingConfig(remove_precursor_window_da=None),
    )
    index = RepresentativeEntropyIndex(library)
    query = _spectrum("SAMESTRUCTURE1", "query", "CCO", mass, [10, 20], [1, 1])

    hits = index.search_molecule(
        [query],
        mass_window_da=5,
        top_n=10,
        exclude_inchikey14={"SAMESTRUCTURE1"},
    )

    assert all(hit.inchikey14 != "SAMESTRUCTURE1" for hit in hits)


def test_representative_index_can_require_matching_polarity() -> None:
    mass = exact_mass("CCO")
    library = CompactSpectralLibraryIndex(
        [_spectrum("POSITIVEONLY1", "positive", "CCO", mass, [10, 20], [1, 1])],
        preprocessing=SpectrumPreprocessingConfig(remove_precursor_window_da=None),
    )
    index = RepresentativeEntropyIndex(library)
    query = _spectrum("NEGATIVEQUERY1", "negative", "CCN", mass, [10, 20], [1, 1])
    query.ionization_mode = "negative"

    unrestricted = index.search_molecule([query], top_n=1)
    restricted = index.search_molecule([query], top_n=1, require_same_polarity=True)

    assert unrestricted
    assert restricted == []


def test_raw_representative_index_uses_linear_richest_spectrum(tmp_path) -> None:
    reference_mass = exact_mass("CCO")
    query_mass = reference_mass + 14.0
    path = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "inchikey14": ["REFERENCEKEY1", "REFERENCEKEY1", "DISTRACTORKEY"],
            "normalized_smiles": ["CCO", "CCO", "CCN"],
            "precursor_mz": [
                theoretical_precursor_mz(reference_mass, "[M+H]+"),
                theoretical_precursor_mz(reference_mass, "[M+H]+"),
                theoretical_precursor_mz(query_mass - 5, "[M+H]+"),
            ],
            "adduct": ["[M+H]+"] * 3,
            "ionization_mode": ["positive"] * 3,
            "num_peaks": [1, 3, 2],
            "ms2_mzs": [[10.0], [10.0, 20.0, 30.0], [8.0, 17.0]],
            "ms2_normalized_intensities": [[1.0], [1.0, 0.8, 0.004], [1.0, 1.0]],
        }
    ).to_parquet(path, index=False)

    index = RawRepresentativeEntropyIndex.from_parquet(path, batch_size=2)
    query = _spectrum(
        "QUERYSTRUCTURE", "query", "CCCO", query_mass, [24, 34, 44], [1.0, 0.8, 0.004]
    )
    hits = index.search_molecule([query], mass_window_da=30, top_n=2)

    assert len(index) == 2
    assert index.peak_count == 5
    assert hits[0].inchikey14 == "REFERENCEKEY1"
    assert hits[0].library_spectrum_index == 1
    assert hits[0].similarity > 0.99


def test_raw_representative_index_retains_and_collapses_polarities(tmp_path) -> None:
    mass = exact_mass("CCO")
    path = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "inchikey14": ["REFERENCEKEY12", "REFERENCEKEY12", "DISTRACTORKEY"],
            "normalized_smiles": ["CCO", "CCO", "CCN"],
            "precursor_mz": [
                theoretical_precursor_mz(mass, "[M+H]+"),
                theoretical_precursor_mz(mass, "[M-H]-"),
                theoretical_precursor_mz(mass + 1, "[M-H]-"),
            ],
            "adduct": ["[M+H]+", "[M-H]-", "[M-H]-"],
            "ionization_mode": ["positive", "negative", "negative"],
            "num_peaks": [3, 2, 2],
            "ms2_mzs": [[10.0, 20.0, 30.0], [40.0, 50.0], [15.0, 25.0]],
            "ms2_normalized_intensities": [[1.0, 0.8, 0.6], [1.0, 0.7], [1.0, 1.0]],
        }
    ).to_parquet(path, index=False)

    index = RawRepresentativeEntropyIndex.from_parquet(
        path,
        batch_size=2,
        per_polarity=True,
    )
    query = _spectrum("QUERYSTRUCTURE", "negative", "CCCO", mass, [40, 50], [1.0, 0.7])
    query.ionization_mode = "negative"
    hits = index.search_molecule(
        [query],
        mass_window_da=5,
        top_n=10,
        require_same_polarity=True,
    )

    assert len(index) == 3
    assert index.structure_count == 2
    assert hits[0].inchikey14 == "REFERENCEKEY12"
    assert sum(hit.inchikey14 == "REFERENCEKEY12" for hit in hits) == 1
    excluded = index.search_molecule(
        [query],
        mass_window_da=5,
        top_n=10,
        exclude_inchikey14={"REFERENCEKEY12"},
    )
    assert all(hit.inchikey14 != "REFERENCEKEY12" for hit in excluded)
