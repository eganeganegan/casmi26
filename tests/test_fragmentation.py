import numpy as np
import pandas as pd

from casmi.ranking import (
    FRAGMENT_FEATURES,
    FragmentationConfig,
    score_candidate_fragments,
    theoretical_fragment_masses,
)
from casmi.spectra import Spectrum
from scripts.score_fragment_candidates import _parquet_query_spectra


def test_theoretical_fragments_are_bounded_and_cached() -> None:
    products, losses = theoretical_fragment_masses("CCO")
    assert len(products) >= 2
    assert np.all(products >= 15.0)
    assert np.all(losses > 0)
    repeated, _ = theoretical_fragment_masses("CCO")
    np.testing.assert_array_equal(products, repeated)


def test_matching_candidate_has_fragment_signal() -> None:
    products, _ = theoretical_fragment_masses("CCO")
    spectrum = Spectrum(
        molecule_id="q",
        spectrum_id="s",
        precursor_mz=47.0491,
        adduct="[M+H]+",
        collision_energy=20.0,
        mz=products[: min(4, len(products))],
        intensity=np.linspace(1.0, 0.4, min(4, len(products))),
        ionization_mode="positive",
    )
    config = FragmentationConfig(mz_tolerance_da=0.01)
    matching = score_candidate_fragments("CCO", [spectrum], config)
    unrelated = score_candidate_fragments("c1ccccc1", [spectrum], config)
    assert set(matching) == set(FRAGMENT_FEATURES)
    assert matching["fragment_scored"] == 1.0
    assert matching["fragment_intensity_fraction_max"] > 0.0
    assert matching["fragment_intensity_fraction_max"] > unrelated[
        "fragment_intensity_fraction_max"
    ]


def test_empty_spectra_returns_unscored_features() -> None:
    values = score_candidate_fragments("CCO", [])
    assert set(values) == set(FRAGMENT_FEATURES)
    assert not any(values.values())


def test_raw_parquet_query_loader_removes_structure_labels(tmp_path) -> None:
    path = tmp_path / "test.parquet"
    pd.DataFrame(
        {
            "molecule_id": ["q"],
            "spectrum_id": ["s"],
            "precursor_mz": [47.0],
            "adduct": ["[M+H]+"],
            "collision_energy_ev": [20.0],
            "ionization_mode": ["positive"],
            "ms2_mzs": [[16.0, 32.0]],
            "ms2_normalized_intensities": [[0.5, 1.0]],
        }
    ).to_parquet(path, index=False)
    loaded = _parquet_query_spectra(path, {"q"}, max_spectra=1)
    assert len(loaded["q"]) == 1
    assert loaded["q"][0].smiles is None
    assert loaded["q"][0].inchikey14 is None
