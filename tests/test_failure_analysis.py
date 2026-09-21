import pandas as pd

from scripts.analyze_candidate_failures import analyze_failures


def test_failure_analysis_separates_database_generation_and_ranking() -> None:
    candidates = pd.DataFrame(
        {
            "molecule_id": ["hit", "hit", "late", "missing"],
            "inchikey14": ["hit", "wrong", "late", "wrong2"],
            "target": [1, 0, 1, 0],
            "final_rank": [1, 2, 30, 1],
            "mass_rank": [2, 1, 30, 1],
        }
    )
    detail, summary = analyze_failures(
        candidates,
        ["hit", "late", "missing", "absent"],
        {"hit", "late", "missing"},
        final_rank_column="final_rank",
    )
    categories = dict(zip(detail["molecule_id"], detail["failure_category"], strict=True))
    assert categories == {
        "hit": "hit_at_25",
        "late": "truth_retrieved_below_25",
        "missing": "truth_missed_by_candidate_generation",
        "absent": "truth_absent_from_database",
    }
    assert summary["database_present"] == 3
