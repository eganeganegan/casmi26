import numpy as np
import pandas as pd
import pytest

from casmi.ranking import (
    FRAGMENT_FEATURES,
    RANKING_FEATURES,
    feature_matrix,
    prepare_ranking_features,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "molecule_id": ["q", "q"],
            "inchikey14": ["LFQSCWFLJHTTHZ", "ATUOYWHBWRKTHZ"],
            "smiles": ["CCO", "CCC"],
            "formula": ["C2H6O", "C3H8"],
            "source": ["coconut", "pubchemlite"],
            "target": [1, 0],
            "rank": [2, 1],
            "mass_rank": [1, 2],
            "mass_error_ppm": [1.0, -2.0],
            "absolute_mass_error_ppm": [1.0, 2.0],
            "mass_score_20ppm": [0.95, 0.90],
            "within_20ppm": [1, 1],
            "within_50ppm": [1, 1],
            "within_1000ppm": [1, 1],
            "predicted_fingerprint_score": [0.7, 0.8],
            "candidate_exact_mass": [46.0, 44.0],
            "neutral_mass": [46.0, 46.0],
            "candidate_count": [2, 2],
            "query_spectrum_count": [2, 2],
            "query_adduct_count": [1, 1],
            "query_positive_fraction": [1.0, 1.0],
            "query_mean_collision_energy": [20.0, 20.0],
        }
    )


def test_prepare_ranking_features_is_finite_and_identity_free() -> None:
    frame = prepare_ranking_features(_candidate_rows())
    matrix = feature_matrix(frame)
    assert tuple(matrix.columns) == RANKING_FEATURES
    assert np.isfinite(matrix.to_numpy()).all()
    forbidden = {
        "target",
        "oracle_fingerprint_score",
        "molecule_id",
        "inchikey14",
        "smiles",
        "formula",
        "source",
    }
    assert forbidden.isdisjoint(matrix.columns)
    assert frame.loc[0, "source_coconut"] == 1
    assert frame.loc[1, "source_pubchemlite"] == 1
    assert not frame.loc[:, FRAGMENT_FEATURES].to_numpy().any()


def test_prepare_ranking_features_rejects_duplicate_candidate() -> None:
    frame = pd.concat([_candidate_rows(), _candidate_rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="repeat"):
        prepare_ranking_features(frame, add_descriptors=False)


def test_prepare_ranking_features_accepts_descriptor_cache() -> None:
    cache = pd.DataFrame(
        {
            "inchikey14": ["LFQSCWFLJHTTHZ", "ATUOYWHBWRKTHZ"],
            "candidate_heteroatom_count": [1.0, 0.0],
            "candidate_ring_count": [0.0, 0.0],
            "candidate_aromatic_fraction": [0.0, 0.0],
            "candidate_logp": [-0.1, 1.4],
        }
    )
    frame = prepare_ranking_features(_candidate_rows(), descriptor_cache=cache)
    assert frame["candidate_heteroatom_count"].tolist() == [1.0, 0.0]


def test_prepare_ranking_features_accepts_prejoined_descriptors() -> None:
    candidates = _candidate_rows()
    candidates.loc[1, "source"] = "chembl37"
    candidates["candidate_heteroatom_count"] = [1.0, 0.0]
    candidates["candidate_ring_count"] = [0.0, 0.0]
    candidates["candidate_aromatic_fraction"] = [0.0, 0.0]
    candidates["candidate_logp"] = [-0.1, 1.4]
    frame = prepare_ranking_features(candidates)
    assert frame["candidate_heteroatom_count"].tolist() == [1.0, 0.0]
    assert frame["source_chembl"].tolist() == [0, 1]
