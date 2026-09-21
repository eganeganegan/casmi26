import pandas as pd

from scripts.predict_ranker import assign_final_ranks


def test_assign_final_ranks_breaks_score_ties_by_mass_rank() -> None:
    frame = pd.DataFrame(
        {
            "molecule_id": ["q", "q", "q", "r"],
            "mass_rank": [3, 1, 2, 1],
            "score": [0.5, 0.5, 0.7, 0.1],
        }
    )
    assert assign_final_ranks(frame, "score").tolist() == [3, 2, 1, 1]
