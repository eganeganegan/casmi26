from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib

from casmi.retrieval import CandidateEvidence, CompactSpectralLibraryIndex, aggregate_hits
from casmi.spectra import Spectrum, SpectrumPreprocessingConfig
from casmi.validation import RankingMetrics, evaluate_mrr


@dataclass(slots=True)
class MoleculePrediction:
    molecule_id: str
    ranked_candidates: list[str]
    evidence: list[CandidateEvidence]


class CASMIPipeline:
    def __init__(
        self,
        *,
        preprocessing: SpectrumPreprocessingConfig | None = None,
        mz_tolerance: float = 0.02,
        mass_tolerance_ppm: float | None = 20.0,
        top_spectra: int = 1000,
        aggregation: str = "top_k_mean",
    ) -> None:
        self.preprocessing = preprocessing or SpectrumPreprocessingConfig()
        self.mz_tolerance = mz_tolerance
        self.mass_tolerance_ppm = mass_tolerance_ppm
        self.top_spectra = top_spectra
        self.aggregation = aggregation
        self.index: CompactSpectralLibraryIndex | None = None

    def fit(self, spectra: Iterable[Spectrum]) -> CASMIPipeline:
        self.index = CompactSpectralLibraryIndex(spectra, self.preprocessing)
        return self

    def predict_molecule(self, spectra: Iterable[Spectrum], max_candidates: int = 25) -> MoleculePrediction:
        if self.index is None:
            raise RuntimeError("Pipeline must be fit or loaded before prediction")
        spectra_list = list(spectra)
        if not spectra_list:
            raise ValueError("No spectra supplied")
        molecule_ids = {spectrum.molecule_id for spectrum in spectra_list}
        if len(molecule_ids) != 1:
            raise ValueError(f"predict_molecule received multiple molecule IDs: {sorted(molecule_ids)}")
        hits = []
        for spectrum in spectra_list:
            hits.extend(
                self.index.search(
                    spectrum,
                    top_n=self.top_spectra,
                    mz_tolerance=self.mz_tolerance,
                    mass_tolerance_ppm=self.mass_tolerance_ppm,
                )
            )
        evidence = aggregate_hits(hits, method=self.aggregation)  # type: ignore[arg-type]
        evidence = evidence[:max_candidates]
        return MoleculePrediction(
            molecule_id=spectra_list[0].molecule_id,
            ranked_candidates=[candidate.smiles for candidate in evidence],
            evidence=evidence,
        )

    def predict_dataset(
        self, spectra: Iterable[Spectrum], max_candidates: int = 25
    ) -> dict[str, MoleculePrediction]:
        grouped: dict[str, list[Spectrum]] = defaultdict(list)
        for spectrum in spectra:
            grouped[spectrum.molecule_id].append(spectrum)
        return {
            molecule_id: self.predict_molecule(group, max_candidates=max_candidates)
            for molecule_id, group in grouped.items()
        }

    def evaluate(
        self, ground_truth: Mapping[str, str], predictions: Mapping[str, MoleculePrediction]
    ) -> RankingMetrics:
        raw = {key: value.ranked_candidates for key, value in predictions.items()}
        return evaluate_mrr(ground_truth, raw)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> CASMIPipeline:
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(value).__name__}")
        return value
