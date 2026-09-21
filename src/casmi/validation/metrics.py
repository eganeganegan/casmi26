from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import median

from casmi.chemistry import smiles_to_inchikey14


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    mrr_at_25: float
    hits_at_1: float
    hits_at_5: float
    hits_at_10: float
    hits_at_25: float
    median_correct_rank: float | None
    fraction_no_correct_candidate: float
    n_queries: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _comparison_key(value: str) -> str:
    stripped = value.strip()
    if len(stripped) == 14 and stripped.isalpha() and stripped.isupper():
        return stripped
    return smiles_to_inchikey14(stripped)


def _candidate_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(";") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def evaluate_mrr(
    ground_truth: Mapping[str, str],
    predictions: Mapping[str, str | Sequence[str]],
    k: int = 25,
) -> RankingMetrics:
    """Evaluate the competition metric after RDKit tautomer/InChIKey14 normalization."""
    if not ground_truth:
        raise ValueError("ground_truth is empty")
    ranks: list[int | None] = []
    for molecule_id, truth in ground_truth.items():
        truth_key = _comparison_key(truth)
        rank: int | None = None
        for raw_rank, raw_candidate in enumerate(
            _candidate_list(predictions.get(molecule_id, []))[:k], start=1
        ):
            try:
                candidate_key = _comparison_key(raw_candidate)
            except (ValueError, RuntimeError):
                continue
            if candidate_key == truth_key:
                rank = raw_rank
                break
        ranks.append(rank)

    n = len(ranks)
    found = [rank for rank in ranks if rank is not None]
    def hit(cutoff: int) -> float:
        return sum(rank is not None and rank <= cutoff for rank in ranks) / n
    return RankingMetrics(
        mrr_at_25=sum(1.0 / rank for rank in found if rank <= k) / n,
        hits_at_1=hit(1),
        hits_at_5=hit(5),
        hits_at_10=hit(10),
        hits_at_25=hit(25),
        median_correct_rank=float(median(found)) if found else None,
        fraction_no_correct_candidate=(n - len(found)) / n,
        n_queries=n,
    )


def evaluate_candidate_recall(
    ground_truth: Mapping[str, str],
    candidates: Mapping[str, str | Sequence[str]],
    ks: Iterable[int] = (25, 100, 500),
) -> dict[str, float]:
    cutoffs = sorted(set(int(k) for k in ks))
    hits = {k: 0 for k in cutoffs}
    for molecule_id, truth in ground_truth.items():
        truth_key = _comparison_key(truth)
        keys: list[str | None] = []
        for value in _candidate_list(candidates.get(molecule_id, []))[: cutoffs[-1]]:
            try:
                key = _comparison_key(value)
            except (ValueError, RuntimeError):
                key = None
            keys.append(key)
        for cutoff in cutoffs:
            hits[cutoff] += truth_key in keys[:cutoff]
    n = len(ground_truth)
    return {f"recall_at_{k}": hits[k] / n if n else 0.0 for k in cutoffs}
