from __future__ import annotations

import math
from collections.abc import Collection, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import numpy as np

from casmi.chemistry import mass_error_ppm, neutral_mass_from_precursor, smiles_to_inchikey14
from casmi.spectra import (
    Spectrum,
    SpectrumPreprocessingConfig,
    clean_spectrum,
    cosine_similarity,
    modified_cosine,
    neutral_loss_cosine,
)


def _polarity(spectrum: Spectrum) -> str:
    if spectrum.ionization_mode:
        return spectrum.ionization_mode.lower()
    if spectrum.adduct.endswith("+"):
        return "positive"
    if spectrum.adduct.endswith("-"):
        return "negative"
    return "unknown"


def _neutral_mass(spectrum: Spectrum) -> float:
    if spectrum.exact_mass is not None and math.isfinite(spectrum.exact_mass):
        return float(spectrum.exact_mass)
    try:
        return neutral_mass_from_precursor(spectrum.precursor_mz, spectrum.adduct)
    except ValueError:
        return float("nan")


def _collision_energy(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=float)
        return float(np.nanmean(array)) if array.size else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class SpectralHit:
    query_spectrum_id: str
    library_spectrum_id: str
    candidate_smiles: str
    candidate_inchikey14: str
    direct_cosine: float
    modified_cosine: float
    neutral_loss_cosine: float
    matched_peaks: int
    explained_query_intensity: float
    mass_error_ppm: float | None
    mass_delta: float | None
    matching_adduct: bool
    collision_energy_difference: float | None

    @property
    def score(self) -> float:
        # Intentionally transparent until a leakage-safe learned reranker is trained.
        return (
            0.60 * self.direct_cosine
            + 0.25 * self.modified_cosine
            + 0.15 * self.neutral_loss_cosine
        )


@dataclass(slots=True)
class CandidateEvidence:
    smiles: str
    inchikey14: str
    score: float
    max_cosine: float
    mean_top_k_cosine: float
    modified_cosine: float
    neutral_loss_cosine: float
    supporting_spectra: int
    supporting_library_spectra: int
    matched_peaks: int
    explained_query_intensity: float
    mass_error_ppm: float | None
    raw_hits: list[SpectralHit] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, object]:
        output = asdict(self)
        output.pop("raw_hits", None)
        return output


class SpectralLibraryIndex:
    """Mass-prefiltered exact and analog spectral retrieval index."""

    def __init__(
        self,
        spectra: Iterable[Spectrum],
        preprocessing: SpectrumPreprocessingConfig | None = None,
    ) -> None:
        self.preprocessing = preprocessing or SpectrumPreprocessingConfig()
        self.spectra: list[Spectrum] = []
        masses: list[float] = []
        for spectrum in spectra:
            if not spectrum.smiles:
                continue
            mz, intensity = clean_spectrum(
                spectrum.mz,
                spectrum.intensity,
                precursor_mz=spectrum.precursor_mz,
                config=self.preprocessing,
            )
            spectrum.mz, spectrum.intensity = mz, intensity
            spectrum.inchikey14 = spectrum.inchikey14 or smiles_to_inchikey14(spectrum.smiles)
            self.spectra.append(spectrum)
            masses.append(_neutral_mass(spectrum))
        self.neutral_masses = np.asarray(masses, dtype=np.float64)
        finite = np.flatnonzero(np.isfinite(self.neutral_masses))
        order = finite[np.argsort(self.neutral_masses[finite], kind="stable")]
        self.mass_order = order.astype(np.int64)
        self.sorted_masses = self.neutral_masses[order]

    def __len__(self) -> int:
        return len(self.spectra)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> SpectralLibraryIndex:
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(value).__name__}")
        return value

    def _mass_candidates(self, mass: float, tolerance_ppm: float) -> np.ndarray:
        delta = abs(mass) * tolerance_ppm * 1e-6
        left = np.searchsorted(self.sorted_masses, mass - delta, side="left")
        right = np.searchsorted(self.sorted_masses, mass + delta, side="right")
        return self.mass_order[left:right]

    def _nearest_mass_candidates(self, mass: float, count: int) -> np.ndarray:
        if not len(self.sorted_masses):
            return np.empty(0, dtype=np.int64)
        center = int(np.searchsorted(self.sorted_masses, mass))
        half = max(1, count // 2)
        left = max(0, center - half)
        right = min(len(self.sorted_masses), left + count)
        left = max(0, right - count)
        local = self.mass_order[left:right]
        distances = np.abs(self.neutral_masses[local] - mass)
        return local[np.argsort(distances, kind="stable")]

    def search(
        self,
        query: Spectrum,
        *,
        top_n: int = 1000,
        mz_tolerance: float = 0.02,
        tolerance_unit: Literal["da", "ppm"] = "da",
        mass_tolerance_ppm: float | None = 20.0,
        require_same_polarity: bool = True,
        analog_mass_window_da: float | None = None,
        max_comparisons: int | None = None,
        exclude_library_spectrum_ids: Collection[str] | None = None,
    ) -> list[SpectralHit]:
        query_mz, query_int = clean_spectrum(
            query.mz,
            query.intensity,
            precursor_mz=query.precursor_mz,
            config=self.preprocessing,
        )
        query_mass = _neutral_mass(query)
        if mass_tolerance_ppm is not None and np.isfinite(query_mass):
            indices = self._mass_candidates(query_mass, mass_tolerance_ppm)
            if not len(indices):
                # A valid submission still needs candidates when the exact-mass library has no hit.
                indices = self._nearest_mass_candidates(query_mass, max(top_n, 100))
        elif analog_mass_window_da is not None and np.isfinite(query_mass):
            left = np.searchsorted(self.sorted_masses, query_mass - analog_mass_window_da, side="left")
            right = np.searchsorted(self.sorted_masses, query_mass + analog_mass_window_da, side="right")
            indices = self.mass_order[left:right]
        else:
            indices = np.arange(len(self.spectra), dtype=np.int64)
        if max_comparisons is not None and len(indices) > max_comparisons:
            # Deterministic mass-stratified sampling bounds broad analog-search cost.
            positions = np.linspace(0, len(indices) - 1, max_comparisons, dtype=np.int64)
            indices = indices[positions]

        query_polarity = _polarity(query)
        query_ce = _collision_energy(query.collision_energy)
        excluded = set(exclude_library_spectrum_ids or ())
        hits: list[SpectralHit] = []
        for index in indices:
            reference = self.spectra[int(index)]
            if reference.spectrum_id in excluded:
                continue
            if require_same_polarity and _polarity(reference) != query_polarity:
                continue
            direct = cosine_similarity(
                query_mz,
                query_int,
                reference.mz,
                reference.intensity,
                tolerance=mz_tolerance,
                tolerance_unit=tolerance_unit,
            )
            modified = modified_cosine(
                query_mz,
                query_int,
                reference.mz,
                reference.intensity,
                query.precursor_mz,
                reference.precursor_mz,
                tolerance=mz_tolerance,
                tolerance_unit=tolerance_unit,
            )
            losses = neutral_loss_cosine(
                query_mz,
                query_int,
                reference.mz,
                reference.intensity,
                query.precursor_mz,
                reference.precursor_mz,
                tolerance=mz_tolerance,
                tolerance_unit=tolerance_unit,
            )
            reference_mass = self.neutral_masses[int(index)]
            mass_delta = query_mass - reference_mass if np.isfinite(query_mass + reference_mass) else None
            ppm = mass_error_ppm(query_mass, reference_mass) if mass_delta is not None else None
            reference_ce = _collision_energy(reference.collision_energy)
            ce_difference = (
                abs(query_ce - reference_ce)
                if query_ce is not None and reference_ce is not None
                else None
            )
            hits.append(
                SpectralHit(
                    query_spectrum_id=query.spectrum_id,
                    library_spectrum_id=reference.spectrum_id,
                    candidate_smiles=reference.smiles or "",
                    candidate_inchikey14=reference.inchikey14 or "",
                    direct_cosine=direct.score,
                    modified_cosine=modified.score,
                    neutral_loss_cosine=losses.score,
                    matched_peaks=direct.matched_peaks,
                    explained_query_intensity=direct.explained_query_intensity,
                    mass_error_ppm=ppm,
                    mass_delta=mass_delta,
                    matching_adduct=query.adduct == reference.adduct,
                    collision_energy_difference=ce_difference,
                )
            )
        if not hits and require_same_polarity:
            return self.search(
                query,
                top_n=top_n,
                mz_tolerance=mz_tolerance,
                tolerance_unit=tolerance_unit,
                mass_tolerance_ppm=mass_tolerance_ppm,
                require_same_polarity=False,
                analog_mass_window_da=analog_mass_window_da,
                max_comparisons=max_comparisons,
                exclude_library_spectrum_ids=excluded,
            )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_n]


def aggregate_hits(
    hits: Iterable[SpectralHit],
    *,
    method: Literal["max", "mean", "top_k_mean", "noisy_or", "logsumexp"] = "noisy_or",
    top_k: int = 3,
) -> list[CandidateEvidence]:
    grouped: dict[str, list[SpectralHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.candidate_inchikey14, []).append(hit)
    candidates: list[CandidateEvidence] = []
    for key, group in grouped.items():
        scores = np.asarray([np.clip(hit.score, 0.0, 1.0) for hit in group])
        if method == "max":
            aggregate = float(scores.max())
        elif method == "mean":
            aggregate = float(scores.mean())
        elif method == "top_k_mean":
            aggregate = float(np.sort(scores)[-top_k:].mean())
        elif method == "noisy_or":
            # First pool repeats within each query spectrum, then combine independent spectra.
            by_query: dict[str, float] = {}
            for hit in group:
                by_query[hit.query_spectrum_id] = max(by_query.get(hit.query_spectrum_id, 0.0), hit.score)
            aggregate = float(1.0 - np.prod([1.0 - np.clip(v, 0.0, 1.0) for v in by_query.values()]))
        elif method == "logsumexp":
            maximum = float(scores.max())
            aggregate = maximum + float(np.log(np.exp(scores - maximum).sum()))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        best = max(group, key=lambda hit: hit.score)
        top_direct = sorted((hit.direct_cosine for hit in group), reverse=True)[:top_k]
        ppm_values = [abs(hit.mass_error_ppm) for hit in group if hit.mass_error_ppm is not None]
        candidates.append(
            CandidateEvidence(
                smiles=best.candidate_smiles,
                inchikey14=key,
                score=aggregate,
                max_cosine=max(hit.direct_cosine for hit in group),
                mean_top_k_cosine=float(np.mean(top_direct)),
                modified_cosine=max(hit.modified_cosine for hit in group),
                neutral_loss_cosine=max(hit.neutral_loss_cosine for hit in group),
                supporting_spectra=len({hit.query_spectrum_id for hit in group}),
                supporting_library_spectra=len({hit.library_spectrum_id for hit in group}),
                matched_peaks=max(hit.matched_peaks for hit in group),
                explained_query_intensity=max(hit.explained_query_intensity for hit in group),
                mass_error_ppm=min(ppm_values) if ppm_values else None,
                raw_hits=group,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
