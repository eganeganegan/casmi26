import numpy as np
import pandas as pd

from casmi.ranking import (
    ANALOG_RANKING_FEATURES,
    fuse_analog_ranks,
    prepare_analog_ranking_features,
    score_analog_fingerprint_features,
)


def test_analog_features_favor_candidate_related_to_strong_analog() -> None:
    candidates = np.asarray([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    analogs = np.asarray([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)

    features = score_analog_fingerprint_features(candidates, analogs, [0.9, 0.2])

    assert features["analog_score_power4"][0] > features["analog_score_power4"][1]
    assert features["analog_score_linear"][0] > features["analog_score_linear"][1]
    assert features["analog_tanimoto_max"].tolist() == [1.0, 1.0]
    assert features["analog_tanimoto_top"].tolist() == [1.0, 0.0]


def test_analog_features_handle_empty_analogs() -> None:
    features = score_analog_fingerprint_features(
        np.ones((2, 8), dtype=np.uint8),
        np.empty((0, 8), dtype=np.uint8),
        [],
    )

    assert all(values.tolist() == [0.0, 0.0] for values in features.values())


def test_analog_rank_fusion_preserves_baseline_when_analog_scores_tie() -> None:
    frame = pd.DataFrame(
        {
            "molecule_id": ["q", "q", "q"],
            "rank": [1, 2, 3],
            "analog_score_power4": [0.0, 0.0, 0.0],
        }
    )

    fused = fuse_analog_ranks(frame)

    assert fused["rank"].tolist() == [1, 2, 3]
    assert fused["baseline_rank"].tolist() == [1, 2, 3]
    assert fused["analog_rank"].tolist() == [1, 2, 3]


def test_prepare_analog_ranking_features_adds_query_relative_values() -> None:
    frame = pd.DataFrame(
        {
            "molecule_id": ["q", "q"],
            "analog_score_power4": [0.8, 0.2],
            "analog_score_linear": [0.9, 0.3],
            "analog_tanimoto_max": [1.0, 0.5],
            "analog_tanimoto_top": [1.0, 0.2],
            "analog_tanimoto_weighted_mean": [0.7, 0.3],
            "analog_top_similarity": [0.9, 0.9],
        }
    )

    output = prepare_analog_ranking_features(frame)

    assert set(ANALOG_RANKING_FEATURES) <= set(output.columns)
    assert output["analog_rank"].tolist() == [1.0, 2.0]
    assert output["analog_rank_percentile"].tolist() == [0.0, 1.0]
    assert np.allclose(output["analog_score_delta_max"], [0.0, -0.6])
