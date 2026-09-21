import pytest

from casmi.validation import evaluate_candidate_recall, evaluate_mrr


def test_mrr_and_hits_with_structure_normalization() -> None:
    truth = {"a": "CCO", "b": "F[C@H](Cl)Br", "c": "c1ccccc1"}
    predictions = {
        "a": ["CC", "OCC"],
        "b": ["F[C@@H](Cl)Br"],
        "c": ["CCO"],
    }
    result = evaluate_mrr(truth, predictions)
    assert result.mrr_at_25 == pytest.approx((0.5 + 1.0) / 3)
    assert result.hits_at_1 == pytest.approx(1 / 3)
    assert result.hits_at_5 == pytest.approx(2 / 3)
    assert result.fraction_no_correct_candidate == pytest.approx(1 / 3)
    assert result.median_correct_rank == 1.5


def test_duplicate_candidates_consume_official_rank_positions() -> None:
    truth = {"a": "CCO"}
    predictions = {"a": ["CC", "CC", "OCC"]}
    assert evaluate_mrr(truth, predictions).mrr_at_25 == pytest.approx(1 / 3)


def test_candidate_recall() -> None:
    recall = evaluate_candidate_recall({"a": "CCO"}, {"a": ["CC", "CCO"]}, (1, 2, 25))
    assert recall == {"recall_at_1": 0.0, "recall_at_2": 1.0, "recall_at_25": 1.0}
