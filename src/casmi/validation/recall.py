from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import pandas as pd

from casmi.chemistry import smiles_to_inchikey14
from casmi.validation.metrics import evaluate_candidate_recall


def reciprocal_rank_union(
    channels: Mapping[str, Mapping[str, Sequence[str]]], k_constant: int = 60
) -> dict[str, list[str]]:
    """Fuse candidate channels for recall accounting without score-scale assumptions."""
    molecule_ids = set().union(*(values.keys() for values in channels.values())) if channels else set()
    output: dict[str, list[str]] = {}
    for molecule_id in molecule_ids:
        scores: dict[str, float] = defaultdict(float)
        representatives: dict[str, str] = {}
        for values in channels.values():
            seen_in_channel: set[str] = set()
            for rank, smiles in enumerate(values.get(molecule_id, []), start=1):
                try:
                    key = smiles_to_inchikey14(smiles)
                except (ValueError, RuntimeError):
                    continue
                if key in seen_in_channel:
                    continue
                seen_in_channel.add(key)
                representatives.setdefault(key, smiles)
                scores[key] += 1.0 / (k_constant + rank)
        ranked_keys = sorted(scores, key=scores.get, reverse=True)
        output[molecule_id] = [representatives[key] for key in ranked_keys]
    return output


def candidate_recall_report(
    ground_truth: Mapping[str, str],
    channels: Mapping[str, Mapping[str, Sequence[str]]],
    ks: Sequence[int] = (25, 100, 500, 1000, 5000),
) -> pd.DataFrame:
    """Report candidate presence separately for every source and their RRF union."""
    rows: list[dict[str, float | int | str]] = []
    expanded = dict(channels)
    expanded["union"] = reciprocal_rank_union(channels)
    for channel, candidates in expanded.items():
        metrics = evaluate_candidate_recall(ground_truth, candidates, ks)
        for name, value in metrics.items():
            rows.append(
                {
                    "channel": channel,
                    "cutoff": int(name.removeprefix("recall_at_")),
                    "recall": value,
                    "n_molecules": len(ground_truth),
                }
            )
    return pd.DataFrame(rows)


def stratified_candidate_recall(
    ground_truth: Mapping[str, str],
    candidates: Mapping[str, Sequence[str]],
    strata: Mapping[str, Mapping[str, object]],
    *,
    ks: Sequence[int] = (25, 100, 500),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stratum_name, labels in strata.items():
        grouped: dict[str, list[str]] = defaultdict(list)
        for molecule_id in ground_truth:
            grouped[str(labels.get(molecule_id, "unknown"))].append(molecule_id)
        for level, molecule_ids in grouped.items():
            subset_truth = {molecule_id: ground_truth[molecule_id] for molecule_id in molecule_ids}
            subset_candidates = {molecule_id: candidates.get(molecule_id, []) for molecule_id in molecule_ids}
            metrics = evaluate_candidate_recall(subset_truth, subset_candidates, ks)
            for name, value in metrics.items():
                rows.append(
                    {
                        "stratum": stratum_name,
                        "level": level,
                        "cutoff": int(name.removeprefix("recall_at_")),
                        "recall": value,
                        "n_molecules": len(molecule_ids),
                    }
                )
    return pd.DataFrame(rows)
