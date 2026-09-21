from pathlib import Path

import pandas as pd
import pytest

from casmi.inference import make_submission, make_submission_from_ranked_frame, validate_submission


def test_submission_deduplicates_and_limits(tmp_path: Path) -> None:
    output = tmp_path / "submission.csv"
    values = ["CCO", "OCC"] + ["C" * n for n in range(1, 30)]
    frame = make_submission({"m1": values}, output)
    assert output.exists()
    candidates = frame.loc[0, "smiles"].split(";")
    assert len(candidates) <= 25
    assert candidates[0] == "CCO"
    validate_submission(frame, ["m1"])


def test_submission_rejects_duplicate_molecules() -> None:
    frame = pd.DataFrame({"molecule_id": ["m", "m"], "smiles": ["CCO", "CCC"]})
    with pytest.raises(ValueError, match="repeats"):
        validate_submission(frame)


def test_submission_rejects_more_than_25() -> None:
    smiles = ";".join("C" * n for n in range(1, 27))
    frame = pd.DataFrame({"molecule_id": ["m"], "smiles": [smiles]})
    with pytest.raises(ValueError, match="26 candidates"):
        validate_submission(frame)


def test_submission_rejects_invalid_candidate_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 1 and 25"):
        make_submission({"m": ["CCO"]}, tmp_path / "submission.csv", max_candidates=0)


def test_submission_from_ranked_frame_uses_existing_keys(tmp_path: Path) -> None:
    ranked = pd.DataFrame(
        {
            "molecule_id": ["m1", "m1", "m1"],
            "rank": [2, 1, 3],
            "smiles": ["OCC", "CCO", "CCC"],
            "inchikey14": ["LFQSCWFLJHTTHZ", "LFQSCWFLJHTTHZ", "ATUOYWHBWRKTHZ"],
        }
    )
    result = make_submission_from_ranked_frame(ranked, tmp_path / "submission.csv")
    assert result.loc[0, "smiles"] == "CCO;CCC"
