from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from casmi.spectra.preprocessing import bin_spectrum, neutral_loss_spectrum


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    score: float
    matched_peaks: int
    explained_query_intensity: float


def _tolerance(mz: float, tolerance: float, unit: Literal["da", "ppm"]) -> float:
    if unit == "da":
        return tolerance
    if unit == "ppm":
        return abs(mz) * tolerance * 1e-6
    raise ValueError(f"Unknown tolerance unit: {unit}")


def match_peaks(
    query_mz: np.ndarray,
    query_intensity: np.ndarray,
    reference_mz: np.ndarray,
    reference_intensity: np.ndarray,
    *,
    tolerance: float = 0.02,
    tolerance_unit: Literal["da", "ppm"] = "da",
    shifts: tuple[float, ...] = (0.0,),
) -> list[tuple[int, int]]:
    """Match sorted peaks via a window scan and greedy unique maximum-product selection."""
    q_mz = np.asarray(query_mz, dtype=np.float64)
    r_mz = np.asarray(reference_mz, dtype=np.float64)
    q_int = np.asarray(query_intensity, dtype=np.float64)
    r_int = np.asarray(reference_intensity, dtype=np.float64)
    candidates: list[tuple[float, int, int]] = []
    for shift in shifts:
        left = 0
        for qi, target in enumerate(q_mz):
            tol = _tolerance(target, tolerance, tolerance_unit)
            shifted_target = target - shift
            while left < len(r_mz) and r_mz[left] < shifted_target - tol:
                left += 1
            ri = left
            while ri < len(r_mz) and r_mz[ri] <= shifted_target + tol:
                candidates.append((float(q_int[qi] * r_int[ri]), qi, ri))
                ri += 1
    candidates.sort(reverse=True)
    used_q: set[int] = set()
    used_r: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, qi, ri in candidates:
        if qi not in used_q and ri not in used_r:
            used_q.add(qi)
            used_r.add(ri)
            matches.append((qi, ri))
    return matches


def cosine_similarity(
    query_mz: np.ndarray,
    query_intensity: np.ndarray,
    reference_mz: np.ndarray,
    reference_intensity: np.ndarray,
    *,
    tolerance: float = 0.02,
    tolerance_unit: Literal["da", "ppm"] = "da",
    shifts: tuple[float, ...] = (0.0,),
) -> SimilarityResult:
    q_int = np.asarray(query_intensity, dtype=np.float64)
    r_int = np.asarray(reference_intensity, dtype=np.float64)
    q_norm = float(np.linalg.norm(q_int))
    r_norm = float(np.linalg.norm(r_int))
    if q_norm == 0 or r_norm == 0:
        return SimilarityResult(0.0, 0, 0.0)
    matches = match_peaks(
        query_mz,
        q_int,
        reference_mz,
        r_int,
        tolerance=tolerance,
        tolerance_unit=tolerance_unit,
        shifts=shifts,
    )
    dot = sum(q_int[qi] * r_int[ri] for qi, ri in matches)
    total_query = float(np.sum(q_int))
    explained = sum(q_int[qi] for qi, _ in matches) / total_query if total_query else 0.0
    return SimilarityResult(float(dot / (q_norm * r_norm)), len(matches), float(explained))


def modified_cosine(
    query_mz: np.ndarray,
    query_intensity: np.ndarray,
    reference_mz: np.ndarray,
    reference_intensity: np.ndarray,
    query_precursor_mz: float,
    reference_precursor_mz: float,
    *,
    tolerance: float = 0.02,
    tolerance_unit: Literal["da", "ppm"] = "da",
) -> SimilarityResult:
    shift = float(query_precursor_mz) - float(reference_precursor_mz)
    return cosine_similarity(
        query_mz,
        query_intensity,
        reference_mz,
        reference_intensity,
        tolerance=tolerance,
        tolerance_unit=tolerance_unit,
        shifts=(0.0, shift),
    )


def neutral_loss_cosine(
    query_mz: np.ndarray,
    query_intensity: np.ndarray,
    reference_mz: np.ndarray,
    reference_intensity: np.ndarray,
    query_precursor_mz: float,
    reference_precursor_mz: float,
    *,
    tolerance: float = 0.02,
    tolerance_unit: Literal["da", "ppm"] = "da",
) -> SimilarityResult:
    q_loss, q_int = neutral_loss_spectrum(query_precursor_mz, query_mz, query_intensity)
    r_loss, r_int = neutral_loss_spectrum(reference_precursor_mz, reference_mz, reference_intensity)
    return cosine_similarity(
        q_loss,
        q_int,
        r_loss,
        r_int,
        tolerance=tolerance,
        tolerance_unit=tolerance_unit,
    )


def binned_cosine(
    query_mz: np.ndarray,
    query_intensity: np.ndarray,
    reference_mz: np.ndarray,
    reference_intensity: np.ndarray,
    *,
    max_mz: float = 1200.0,
    bin_width: float = 0.1,
) -> float:
    q = bin_spectrum(query_mz, query_intensity, max_mz=max_mz, bin_width=bin_width)
    r = bin_spectrum(reference_mz, reference_intensity, max_mz=max_mz, bin_width=bin_width)
    denom = float(np.linalg.norm(q) * np.linalg.norm(r))
    return float(np.dot(q, r) / denom) if denom else 0.0
