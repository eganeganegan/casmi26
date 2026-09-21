from pathlib import Path

import numpy as np
import polars as pl
import pytest

from casmi.chemistry import exact_mass, morgan_fingerprint, theoretical_precursor_mz
from casmi.models import (
    FingerprintFeatureConfig,
    FingerprintModelConfig,
    SpectrumFingerprintMLP,
    featurize_spectrum,
    fingerprint_compatibility,
    load_fingerprint_model,
    pool_fingerprint_probabilities,
    save_fingerprint_model,
)
from casmi.retrieval import CandidateDatabase, CandidateFingerprintIndex
from casmi.spectra import Spectrum


def test_spectrum_fingerprint_feature_shape_and_metadata() -> None:
    mass = exact_mass("CCO")
    spectrum = Spectrum(
        molecule_id="m",
        spectrum_id="s",
        precursor_mz=theoretical_precursor_mz(mass, "[M+H]+"),
        adduct="[M+H]+",
        collision_energy=20,
        mz=[31.0, 45.0],
        intensity=[1.0, 0.25],
        ionization_mode="positive",
    )
    config = FingerprintFeatureConfig(max_mz=100, bin_width=1.0)
    features = featurize_spectrum(spectrum, feature_config=config)
    assert features.shape == (105,)
    assert features[31] == pytest.approx(1.0)
    assert features[-2:].tolist() == [1.0, 0.0]


def test_probability_pooling_and_compatibility_prefers_truth() -> None:
    truth = morgan_fingerprint("CCO", n_bits=128)
    other = morgan_fingerprint("c1ccccc1", n_bits=128)
    probabilities = truth.astype(np.float32) * 0.9 + (1 - truth) * 0.1
    pooled = pool_fingerprint_probabilities(np.stack([probabilities, probabilities]))
    scores = fingerprint_compatibility(pooled, np.stack([truth, other]))
    assert scores["expected_tanimoto"][0] > scores["expected_tanimoto"][1]
    assert scores["binary_tanimoto"][0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-empty"):
        pool_fingerprint_probabilities(np.empty((0, 128)))


def test_candidate_fingerprint_index_build_rank_and_round_trip(tmp_path: Path) -> None:
    database = CandidateDatabase(
        pl.DataFrame(
            {
                "candidate_id": ["ethanol", "benzene"],
                "smiles": ["CCO", "c1ccccc1"],
                "canonical_smiles": ["CCO", "c1ccccc1"],
                "inchikey14": ["LFQSCWFLJHTTHZ", "UHOVQNZJYSORNB"],
                "formula": ["C2H6O", "C6H6"],
                "exact_mass": [exact_mass("CCO"), exact_mass("c1ccccc1")],
                "source": ["test", "test"],
            }
        )
    )
    index, report = CandidateFingerprintIndex.from_candidate_database(
        database, n_bits=128, radius=2
    )
    assert report == {"input_structures": 2, "indexed_structures": 2, "invalid": 0}
    assert index.packed_fingerprints.shape == (2, 16)
    probabilities = morgan_fingerprint("CCO", n_bits=128).astype(np.float32)
    assert index.rank_probabilities(probabilities, top_n=1)[0].inchikey14 == "LFQSCWFLJHTTHZ"
    assert index.indices_for_keys(["UHOVQNZJYSORNB"]).tolist() == [1]

    output = tmp_path / "fingerprints.joblib"
    index.save(output)
    loaded = CandidateFingerprintIndex.load(output)
    assert loaded.unpack().shape == (2, 128)


def test_fingerprint_mlp_forward_and_round_trip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    feature_config = FingerprintFeatureConfig(max_mz=20, bin_width=1)
    model_config = FingerprintModelConfig(n_bits=32, hidden_dims=(16,), dropout=0)
    model = SpectrumFingerprintMLP(feature_config.input_dim, model_config)
    output = model(torch.zeros((3, feature_config.input_dim)))
    assert tuple(output.shape) == (3, 32)
    path = tmp_path / "model.pt"
    save_fingerprint_model(
        model,
        path,
        feature_config=feature_config,
        model_config=model_config,
        extra={"test": True},
    )
    loaded, loaded_features, loaded_model, extra = load_fingerprint_model(path)
    assert tuple(loaded(torch.zeros((1, loaded_features.input_dim))).shape) == (1, 32)
    assert loaded_model == model_config
    assert extra == {"test": True}
