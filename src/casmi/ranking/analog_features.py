from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

ANALOG_FEATURES = (
    "analog_score_power4",
    "analog_score_linear",
    "analog_tanimoto_max",
    "analog_tanimoto_top",
    "analog_tanimoto_weighted_mean",
    "analog_top_similarity",
)

ANALOG_RANKING_FEATURES = ANALOG_FEATURES + (
    "analog_rank",
    "reciprocal_analog_rank",
    "analog_rank_percentile",
    "analog_score_max",
    "analog_score_delta_max",
    "analog_score_z",
)


def score_analog_fingerprint_features(
    candidate_fingerprints: np.ndarray,
    analog_fingerprints: np.ndarray,
    analog_similarities: Sequence[float] | np.ndarray,
    *,
    similarity_power: float = 4.0,
) -> dict[str, np.ndarray]:
    """Propagate spectral-analog evidence through structural Tanimoto similarity."""
    candidates = np.asarray(candidate_fingerprints, dtype=np.uint8)
    analogs = np.asarray(analog_fingerprints, dtype=np.uint8)
    similarities = np.clip(np.asarray(analog_similarities, dtype=np.float32), 0.0, None)
    if candidates.ndim != 2 or analogs.ndim != 2:
        raise ValueError("candidate_fingerprints and analog_fingerprints must be matrices")
    if candidates.shape[1] != analogs.shape[1]:
        raise ValueError("Candidate and analog fingerprints must have the same width")
    if len(analogs) != len(similarities):
        raise ValueError("Each analog fingerprint must have one spectral similarity")
    candidate_count = len(candidates)
    zeros = np.zeros(candidate_count, dtype=np.float32)
    if not candidate_count or not len(analogs):
        return {feature: zeros.copy() for feature in ANALOG_FEATURES}

    candidate_bits = candidates.sum(axis=1, dtype=np.int32)
    powered_weights = np.power(similarities, similarity_power)
    score_power = zeros.copy()
    score_linear = zeros.copy()
    tanimoto_max = zeros.copy()
    tanimoto_top = zeros.copy()
    weighted_sum = zeros.copy()
    for analog_index, (analog, similarity, powered_weight) in enumerate(
        zip(analogs, similarities, powered_weights, strict=True)
    ):
        set_bits = np.flatnonzero(analog)
        intersection = (
            candidates[:, set_bits].sum(axis=1, dtype=np.int32)
            if len(set_bits)
            else np.zeros(candidate_count, dtype=np.int32)
        )
        union = candidate_bits + len(set_bits) - intersection
        tanimoto = np.divide(
            intersection,
            union,
            out=np.ones(candidate_count, dtype=np.float32),
            where=union > 0,
        )
        np.maximum(score_power, tanimoto * powered_weight, out=score_power)
        np.maximum(score_linear, tanimoto * similarity, out=score_linear)
        np.maximum(tanimoto_max, tanimoto, out=tanimoto_max)
        weighted_sum += tanimoto * powered_weight
        if analog_index == 0:
            tanimoto_top[:] = tanimoto
    weight_total = float(powered_weights.sum())
    weighted_mean = weighted_sum / weight_total if weight_total > 0.0 else zeros.copy()
    return {
        "analog_score_power4": score_power,
        "analog_score_linear": score_linear,
        "analog_tanimoto_max": tanimoto_max,
        "analog_tanimoto_top": tanimoto_top,
        "analog_tanimoto_weighted_mean": weighted_mean.astype(np.float32, copy=False),
        "analog_top_similarity": np.full(
            candidate_count, float(similarities[0]), dtype=np.float32
        ),
    }


def fuse_analog_ranks(
    frame: pd.DataFrame,
    *,
    analog_weight: float = 0.1,
    rrf_k: float = 60.0,
    baseline_rank_column: str = "rank",
) -> pd.DataFrame:
    """Fuse a frozen baseline rank with analog evidence using weighted RRF."""
    required = {"molecule_id", baseline_rank_column, "analog_score_power4"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Analog fusion frame is missing columns: {sorted(missing)}")
    if not 0.0 <= analog_weight <= 1.0:
        raise ValueError("analog_weight must be between 0 and 1")
    if rrf_k < 0.0:
        raise ValueError("rrf_k must be non-negative")
    output = frame.copy()
    output["baseline_rank"] = output[baseline_rank_column].astype(int)
    analog_order = output.sort_values(
        ["molecule_id", "analog_score_power4", "baseline_rank"],
        ascending=[True, False, True],
        kind="stable",
    )
    analog_rank = pd.Series(index=output.index, dtype=np.int64)
    analog_rank.loc[analog_order.index] = (
        analog_order.groupby("molecule_id", sort=False).cumcount().to_numpy() + 1
    )
    output["analog_rank"] = analog_rank.astype(int)
    output["analog_fused_score"] = (
        (1.0 - analog_weight) / (rrf_k + output["baseline_rank"].astype(float))
        + analog_weight / (rrf_k + output["analog_rank"].astype(float))
    )
    fused_order = output.sort_values(
        ["molecule_id", "analog_fused_score", "baseline_rank"],
        ascending=[True, False, True],
        kind="stable",
    )
    final_rank = pd.Series(index=output.index, dtype=np.int64)
    final_rank.loc[fused_order.index] = (
        fused_order.groupby("molecule_id", sort=False).cumcount().to_numpy() + 1
    )
    output[baseline_rank_column] = final_rank.astype(int)
    return output.sort_values(["molecule_id", baseline_rank_column], kind="stable")


def prepare_analog_ranking_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add query-relative ranks and normalizations to raw analog evidence."""
    required = {"molecule_id", *ANALOG_FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Analog feature frame is missing columns: {sorted(missing)}")
    output = frame.copy()
    order = output.sort_values(
        ["molecule_id", "analog_score_power4"],
        ascending=[True, False],
        kind="stable",
    )
    ranks = pd.Series(index=output.index, dtype=np.int64)
    ranks.loc[order.index] = order.groupby("molecule_id", sort=False).cumcount().to_numpy() + 1
    output["analog_rank"] = ranks.astype(np.float32)
    output["reciprocal_analog_rank"] = 1.0 / output["analog_rank"]
    counts = output.groupby("molecule_id")["molecule_id"].transform("size").astype(float)
    output["analog_rank_percentile"] = np.divide(
        output["analog_rank"] - 1.0,
        np.maximum(counts - 1.0, 1.0),
    )
    grouped = output.groupby("molecule_id")["analog_score_power4"]
    score_max = grouped.transform("max")
    score_mean = grouped.transform("mean")
    score_std = grouped.transform("std").fillna(0.0)
    output["analog_score_max"] = score_max
    output["analog_score_delta_max"] = output["analog_score_power4"] - score_max
    output["analog_score_z"] = np.divide(
        output["analog_score_power4"] - score_mean,
        score_std,
        out=np.zeros(len(output), dtype=np.float64),
        where=score_std.to_numpy() > 0.0,
    )
    for column in ANALOG_RANKING_FEATURES:
        output[column] = (
            pd.to_numeric(output[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )
    return output
