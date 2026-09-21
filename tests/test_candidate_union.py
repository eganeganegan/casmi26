import pandas as pd
import pytest

from casmi.retrieval import prune_wide_fallback_candidates


def test_prune_wide_fallback_keeps_primary_and_bounded_fingerprint_rows() -> None:
    frame = pd.DataFrame(
        {
            "molecule_id": ["q", "q", "q"],
            "inchikey14": ["primary", "fallback", "drop"],
            "within_20ppm": [1, 0, 0],
            "wide_mass_rank": [1, 3, 2],
            "wide_fingerprint_rank": [100, 2, 11],
            "absolute_mass_error_ppm": [1.0, 50.0, 100.0],
            "predicted_fingerprint_score": [0.2, 0.9, 0.8],
            "candidate_count": [3, 3, 3],
        }
    )
    output = prune_wide_fallback_candidates(frame, max_wide_fingerprint_rank=10)
    assert set(output["inchikey14"]) == {"primary", "fallback"}
    assert output["candidate_count"].tolist() == [2, 2]
    assert output.sort_values("rank").iloc[0]["inchikey14"] == "fallback"
    assert output.sort_values("mass_rank").iloc[0]["inchikey14"] == "primary"


def test_prune_wide_fallback_rejects_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "molecule_id": ["q", "q"],
            "inchikey14": ["same", "same"],
            "within_20ppm": [1, 0],
            "wide_mass_rank": [1, 2],
            "wide_fingerprint_rank": [1, 2],
            "absolute_mass_error_ppm": [1.0, 2.0],
            "predicted_fingerprint_score": [0.5, 0.4],
        }
    )
    with pytest.raises(ValueError, match="repeat"):
        prune_wide_fallback_candidates(frame, max_wide_fingerprint_rank=10)
