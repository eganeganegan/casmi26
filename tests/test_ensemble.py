import pytest

from casmi.ranking import fuse_rankings, fuse_scores


def test_rrf_favors_candidate_supported_by_multiple_channels() -> None:
    fused = fuse_rankings(
        {"mass": ["a", "b", "c"], "fingerprint": ["b", "d", "a"]},
        method="rrf",
        rrf_k=60,
    )
    assert fused[0].candidate == "b"
    assert len({candidate.candidate for candidate in fused}) == len(fused)


def test_borda_respects_channel_weights_and_limit() -> None:
    fused = fuse_rankings(
        {"strong": ["a", "b"], "weak": ["b", "a"]},
        method="borda",
        weights={"strong": 2.0, "weak": 0.5},
        limit=1,
    )
    assert [candidate.candidate for candidate in fused] == ["a"]


def test_rank_fusion_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        fuse_rankings({"bad": ["a"]}, weights={"bad": -1.0})


def test_weighted_score_fusion_normalizes_channels() -> None:
    fused = fuse_scores(
        {
            "large_scale": {"a": 100.0, "b": 90.0},
            "small_scale": {"a": 0.1, "b": 0.9},
        },
        normalization="minmax",
        weights={"large_scale": 1.0, "small_scale": 2.0},
    )
    assert fused[0].candidate == "b"
