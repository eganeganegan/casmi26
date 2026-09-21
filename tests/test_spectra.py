import numpy as np
import pytest

from casmi.spectra import (
    SpectrumPreprocessingConfig,
    binned_cosine,
    clean_spectrum,
    cosine_similarity,
    modified_cosine,
    neutral_loss_cosine,
)
from casmi.spectra.preprocessing import parse_peak_array


def test_clean_spectrum_filters_sorts_thresholds_and_caps() -> None:
    mz, intensity = clean_spectrum(
        [50, np.nan, -1, 10, 30, 20, 150],
        [1, 1, 1, -1, 0.001, 0.5, 1],
        precursor_mz=100,
        config=SpectrumPreprocessingConfig(
            min_relative_intensity=0.01,
            top_n=2,
            remove_precursor_window_da=2,
            intensity_transform="raw",
        ),
    )
    np.testing.assert_allclose(mz, [20, 50])
    np.testing.assert_allclose(intensity, [0.5, 1.0])


def test_string_encoded_peak_arrays() -> None:
    np.testing.assert_allclose(parse_peak_array("[1.0, 2.0, 3.0]"), [1, 2, 3])
    np.testing.assert_allclose(parse_peak_array("[1.0 2.0 3.0]"), [1, 2, 3])


def test_cosine_identity() -> None:
    mz = np.array([10.0, 20.0, 30.0])
    intensity = np.array([0.2, 1.0, 0.5])
    result = cosine_similarity(mz, intensity, mz, intensity)
    assert result.score == pytest.approx(1.0)
    assert result.matched_peaks == 3
    assert result.explained_query_intensity == pytest.approx(1.0)
    assert binned_cosine(mz, intensity, mz, intensity) == pytest.approx(1.0)


def test_tolerance_and_unique_peak_matching() -> None:
    result = cosine_similarity(
        np.array([100.0, 100.01]),
        np.array([1.0, 0.5]),
        np.array([100.005]),
        np.array([1.0]),
        tolerance=0.02,
    )
    assert result.matched_peaks == 1


def test_modified_cosine_aligns_precursor_shift() -> None:
    direct = cosine_similarity(
        np.array([50.0, 75.0]), np.ones(2), np.array([60.0, 85.0]), np.ones(2)
    )
    shifted = modified_cosine(
        np.array([50.0, 75.0]),
        np.ones(2),
        np.array([60.0, 85.0]),
        np.ones(2),
        110.0,
        120.0,
    )
    assert direct.score == 0
    assert shifted.score == pytest.approx(1.0)


def test_neutral_loss_cosine() -> None:
    result = neutral_loss_cosine(
        np.array([80.0, 90.0]),
        np.ones(2),
        np.array([100.0, 110.0]),
        np.ones(2),
        120.0,
        140.0,
    )
    assert result.score == pytest.approx(1.0)
