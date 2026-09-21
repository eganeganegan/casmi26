from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    candidate: str
    score: float


def fuse_rankings(
    rankings: Mapping[str, Sequence[str]],
    *,
    method: Literal["rrf", "borda"] = "rrf",
    weights: Mapping[str, float] | None = None,
    rrf_k: float = 60.0,
    limit: int | None = None,
) -> list[FusedCandidate]:
    """Fuse deduplicated per-channel rankings with deterministic tie breaking."""
    if method not in {"rrf", "borda"}:
        raise ValueError(f"Unknown rank-fusion method: {method}")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    channel_weights = weights or {}
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    position = 0
    for channel, raw_candidates in rankings.items():
        weight = float(channel_weights.get(channel, 1.0))
        if weight < 0:
            raise ValueError("Channel weights must be non-negative")
        candidates = list(dict.fromkeys(str(candidate) for candidate in raw_candidates))
        length = max(1, len(candidates))
        for rank, candidate in enumerate(candidates, start=1):
            if candidate not in first_seen:
                first_seen[candidate] = position
                position += 1
            contribution = (
                weight / (rrf_k + rank)
                if method == "rrf"
                else weight * (length - rank + 1) / length
            )
            scores[candidate] = scores.get(candidate, 0.0) + contribution
    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key], key))
    if limit is not None:
        ordered = ordered[:limit]
    return [FusedCandidate(candidate, scores[candidate]) for candidate in ordered]


def fuse_scores(
    channel_scores: Mapping[str, Mapping[str, float]],
    *,
    weights: Mapping[str, float] | None = None,
    normalization: Literal["zscore", "minmax", "none"] = "zscore",
    limit: int | None = None,
) -> list[FusedCandidate]:
    """Fuse raw channel scores after per-query channel normalization."""
    if normalization not in {"zscore", "minmax", "none"}:
        raise ValueError(f"Unknown score normalization: {normalization}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    channel_weights = weights or {}
    fused: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    position = 0
    for channel, raw_scores in channel_scores.items():
        weight = float(channel_weights.get(channel, 1.0))
        if weight < 0:
            raise ValueError("Channel weights must be non-negative")
        valid = {
            str(candidate): float(score)
            for candidate, score in raw_scores.items()
            if math.isfinite(float(score))
        }
        if not valid:
            continue
        candidates = list(valid)
        values = np.asarray([valid[candidate] for candidate in candidates], dtype=np.float64)
        if normalization == "zscore":
            scale = float(values.std())
            normalized = (values - values.mean()) / scale if scale > 0 else np.zeros_like(values)
        elif normalization == "minmax":
            scale = float(values.max() - values.min())
            normalized = (values - values.min()) / scale if scale > 0 else np.zeros_like(values)
        else:
            normalized = values
        for candidate, score in zip(candidates, normalized, strict=True):
            if candidate not in first_seen:
                first_seen[candidate] = position
                position += 1
            fused[candidate] = fused.get(candidate, 0.0) + weight * float(score)
    ordered = sorted(fused, key=lambda key: (-fused[key], first_seen[key], key))
    if limit is not None:
        ordered = ordered[:limit]
    return [FusedCandidate(candidate, fused[candidate]) for candidate in ordered]
