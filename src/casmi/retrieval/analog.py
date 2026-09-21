from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from casmi.chemistry import morgan_fingerprint, neutral_mass_from_precursor, tanimoto
from casmi.retrieval.candidate_db import CandidateDatabase
from casmi.retrieval.compact_library import CompactSpectralLibraryIndex
from casmi.retrieval.library import SpectralHit, SpectralLibraryIndex
from casmi.spectra import Spectrum


@dataclass(frozen=True, slots=True)
class PropagatedCandidate:
    smiles: str
    inchikey14: str
    exact_mass: float
    source: str
    mass_error_ppm: float
    mass_score: float
    analog_score: float
    analog_tanimoto: float
    best_analog_inchikey14: str | None
    final_score: float


def molecule_neutral_mass(spectra: Iterable[Spectrum]) -> float:
    masses: list[float] = []
    for spectrum in spectra:
        try:
            mass = neutral_mass_from_precursor(spectrum.precursor_mz, spectrum.adduct)
        except ValueError:
            continue
        if math.isfinite(mass) and mass > 0:
            masses.append(mass)
    if not masses:
        raise ValueError("No spectrum has a supported adduct and finite neutral mass")
    return float(np.median(masses))


class AnalogRetriever:
    """Broad mass-window spectral retrieval kept separate from candidate propagation."""

    def __init__(
        self,
        index: SpectralLibraryIndex | CompactSpectralLibraryIndex,
        *,
        mass_window_da: float = 200.0,
        max_comparisons: int = 5000,
        top_hits: int = 100,
        mz_tolerance: float = 0.02,
    ) -> None:
        self.index = index
        self.mass_window_da = mass_window_da
        self.max_comparisons = max_comparisons
        self.top_hits = top_hits
        self.mz_tolerance = mz_tolerance

    def search_molecule(self, spectra: Iterable[Spectrum]) -> list[SpectralHit]:
        hits: list[SpectralHit] = []
        for spectrum in spectra:
            hits.extend(
                self.index.search(
                    spectrum,
                    top_n=self.top_hits,
                    mz_tolerance=self.mz_tolerance,
                    mass_tolerance_ppm=None,
                    analog_mass_window_da=self.mass_window_da,
                    max_comparisons=self.max_comparisons,
                )
            )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)


def propagate_analog_candidates(
    spectra: Iterable[Spectrum],
    database: CandidateDatabase,
    analog_hits: Iterable[SpectralHit],
    *,
    mass_tolerance_ppm: float = 20.0,
    max_candidates: int = 5000,
    max_analogs: int = 25,
) -> list[PropagatedCandidate]:
    """Rank exact-mass DB candidates by structural proximity to spectral analogs."""
    spectra_list = list(spectra)
    query_mass = molecule_neutral_mass(spectra_list)
    candidates = database.query_mass(query_mass, mass_tolerance_ppm, limit=max_candidates)
    unique_analogs: list[SpectralHit] = []
    seen: set[str] = set()
    for hit in sorted(analog_hits, key=lambda item: item.score, reverse=True):
        if hit.candidate_inchikey14 not in seen:
            seen.add(hit.candidate_inchikey14)
            unique_analogs.append(hit)
        if len(unique_analogs) >= max_analogs:
            break
    analog_fingerprints = [
        (hit, morgan_fingerprint(hit.candidate_smiles)) for hit in unique_analogs
    ]

    output: list[PropagatedCandidate] = []
    for candidate in candidates:
        ppm_error = abs(float(candidate.mass_error_ppm or 0.0))
        mass_score = math.exp(-ppm_error / max(mass_tolerance_ppm, 1e-12))
        best_score = 0.0
        best_similarity = 0.0
        best_key: str | None = None
        if analog_fingerprints:
            candidate_fp = morgan_fingerprint(candidate.smiles)
            for hit, analog_fp in analog_fingerprints:
                similarity = tanimoto(candidate_fp, analog_fp)
                score = hit.score * similarity
                if score > best_score:
                    best_score = score
                    best_similarity = similarity
                    best_key = hit.candidate_inchikey14
        final_score = 0.85 * best_score + 0.15 * mass_score
        output.append(
            PropagatedCandidate(
                smiles=candidate.smiles,
                inchikey14=candidate.inchikey14,
                exact_mass=candidate.exact_mass,
                source=candidate.source,
                mass_error_ppm=float(candidate.mass_error_ppm or 0.0),
                mass_score=mass_score,
                analog_score=best_score,
                analog_tanimoto=best_similarity,
                best_analog_inchikey14=best_key,
                final_score=final_score,
            )
        )
    return sorted(output, key=lambda candidate: candidate.final_score, reverse=True)
