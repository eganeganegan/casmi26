from __future__ import annotations

import math
from array import array
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Literal

import joblib
import numpy as np

from casmi.chemistry import mass_error_ppm, smiles_to_inchikey14
from casmi.retrieval.library import SpectralHit, _collision_energy, _neutral_mass, _polarity
from casmi.spectra import (
    Spectrum,
    SpectrumPreprocessingConfig,
    clean_spectrum,
    cosine_similarity,
    modified_cosine,
    neutral_loss_cosine,
)


class _PackedStrings:
    """UTF-8 strings stored as one byte buffer plus offsets."""

    def __init__(self, data: np.ndarray, offsets: np.ndarray) -> None:
        self.data = np.asarray(data, dtype=np.uint8)
        self.offsets = np.asarray(offsets, dtype=np.uint64)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: int) -> str:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.data[start:end].tobytes().decode("utf-8")


class _PackedStringBuilder:
    def __init__(self) -> None:
        self.data = bytearray()
        self.offsets = array("Q", [0])

    def append(self, value: str) -> None:
        self.data.extend(value.encode("utf-8"))
        self.offsets.append(len(self.data))

    def finish(self) -> _PackedStrings:
        return _PackedStrings(
            np.frombuffer(self.data, dtype=np.uint8).copy(),
            np.frombuffer(self.offsets, dtype=np.uint64).copy(),
        )


def _append_float32(values: array[float], source: np.ndarray) -> None:
    contiguous = np.asarray(source, dtype=np.float32)
    values.frombytes(contiguous.tobytes())


class CompactSpectralLibraryIndex:
    """Array-backed spectral index suitable for a multi-million-spectrum library.

    Peaks use CSR-style offsets, repeated structures and adducts use integer codes,
    and arbitrary identifiers use packed UTF-8 buffers. No ``Spectrum`` objects are
    retained after construction.
    """

    def __init__(
        self,
        spectra: Iterable[Spectrum],
        preprocessing: SpectrumPreprocessingConfig | None = None,
    ) -> None:
        self.preprocessing = preprocessing or SpectrumPreprocessingConfig()
        peak_mz = array("f")
        peak_intensity = array("f")
        peak_offsets = array("Q", [0])
        precursor_mz = array("d")
        neutral_masses = array("d")
        collision_energies = array("f")
        polarities = array("b")
        structure_codes = array("I")
        adduct_codes = array("H")
        spectrum_ids = _PackedStringBuilder()
        structure_smiles = _PackedStringBuilder()
        structure_keys = _PackedStringBuilder()
        adduct_values = _PackedStringBuilder()
        structure_lookup: dict[str, int] = {}
        adduct_lookup: dict[str, int] = {}

        for spectrum in spectra:
            if not spectrum.smiles:
                continue
            mz, intensity = clean_spectrum(
                spectrum.mz,
                spectrum.intensity,
                precursor_mz=spectrum.precursor_mz,
                config=self.preprocessing,
            )
            key = spectrum.inchikey14 or smiles_to_inchikey14(spectrum.smiles)
            structure_code = structure_lookup.get(key)
            if structure_code is None:
                structure_code = len(structure_lookup)
                structure_lookup[key] = structure_code
                structure_smiles.append(spectrum.smiles)
                structure_keys.append(key)
            adduct_code = adduct_lookup.get(spectrum.adduct)
            if adduct_code is None:
                adduct_code = len(adduct_lookup)
                if adduct_code > np.iinfo(np.uint16).max:
                    raise ValueError("Too many distinct adduct labels for uint16 encoding")
                adduct_lookup[spectrum.adduct] = adduct_code
                adduct_values.append(spectrum.adduct)

            _append_float32(peak_mz, mz)
            _append_float32(peak_intensity, intensity)
            peak_offsets.append(len(peak_mz))
            precursor_mz.append(float(spectrum.precursor_mz))
            neutral_masses.append(_neutral_mass(spectrum))
            ce = _collision_energy(spectrum.collision_energy)
            collision_energies.append(float("nan") if ce is None else ce)
            polarity = _polarity(spectrum)
            polarities.append(1 if polarity == "positive" else -1 if polarity == "negative" else 0)
            structure_codes.append(structure_code)
            adduct_codes.append(adduct_code)
            spectrum_ids.append(spectrum.spectrum_id)

        self.peak_mz = np.frombuffer(peak_mz, dtype=np.float32).copy()
        self.peak_intensity = np.frombuffer(peak_intensity, dtype=np.float32).copy()
        self.peak_offsets = np.frombuffer(peak_offsets, dtype=np.uint64).copy()
        self.precursor_mz = np.frombuffer(precursor_mz, dtype=np.float64).copy()
        self.neutral_masses = np.frombuffer(neutral_masses, dtype=np.float64).copy()
        self.collision_energies = np.frombuffer(collision_energies, dtype=np.float32).copy()
        self.polarities = np.frombuffer(polarities, dtype=np.int8).copy()
        self.structure_codes = np.frombuffer(structure_codes, dtype=np.uint32).copy()
        self.adduct_codes = np.frombuffer(adduct_codes, dtype=np.uint16).copy()
        self.spectrum_ids = spectrum_ids.finish()
        self.structure_smiles = structure_smiles.finish()
        self.structure_keys = structure_keys.finish()
        self.adduct_values = adduct_values.finish()
        finite = np.flatnonzero(np.isfinite(self.neutral_masses))
        self.mass_order = finite[
            np.argsort(self.neutral_masses[finite], kind="stable")
        ].astype(np.int64)
        self.sorted_masses = self.neutral_masses[self.mass_order]

    def __len__(self) -> int:
        return len(self.precursor_mz)

    @property
    def peak_count(self) -> int:
        return len(self.peak_mz)

    @property
    def storage_nbytes(self) -> int:
        """Bytes occupied by the index's NumPy buffers (excluding small Python metadata)."""
        arrays = (
            self.peak_mz,
            self.peak_intensity,
            self.peak_offsets,
            self.precursor_mz,
            self.neutral_masses,
            self.collision_energies,
            self.polarities,
            self.structure_codes,
            self.adduct_codes,
            self.mass_order,
            self.sorted_masses,
            self.spectrum_ids.data,
            self.spectrum_ids.offsets,
            self.structure_smiles.data,
            self.structure_smiles.offsets,
            self.structure_keys.data,
            self.structure_keys.offsets,
            self.adduct_values.data,
            self.adduct_values.offsets,
        )
        return sum(array.nbytes for array in arrays)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> CompactSpectralLibraryIndex:
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

    def _peaks(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = int(self.peak_offsets[index]), int(self.peak_offsets[index + 1])
        return self.peak_mz[start:end], self.peak_intensity[start:end]

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
                indices = self._nearest_mass_candidates(query_mass, max(top_n, 100))
        elif analog_mass_window_da is not None and np.isfinite(query_mass):
            left = np.searchsorted(self.sorted_masses, query_mass - analog_mass_window_da, side="left")
            right = np.searchsorted(self.sorted_masses, query_mass + analog_mass_window_da, side="right")
            indices = self.mass_order[left:right]
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if max_comparisons is not None and len(indices) > max_comparisons:
            positions = np.linspace(0, len(indices) - 1, max_comparisons, dtype=np.int64)
            indices = indices[positions]

        query_polarity = _polarity(query)
        query_polarity_code = (
            1 if query_polarity == "positive" else -1 if query_polarity == "negative" else 0
        )
        query_ce = _collision_energy(query.collision_energy)
        excluded = set(exclude_library_spectrum_ids or ())
        hits: list[SpectralHit] = []
        for raw_index in indices:
            index = int(raw_index)
            if self.spectrum_ids[index] in excluded:
                continue
            if require_same_polarity and self.polarities[index] != query_polarity_code:
                continue
            reference_mz, reference_intensity = self._peaks(index)
            direct = cosine_similarity(
                query_mz,
                query_int,
                reference_mz,
                reference_intensity,
                tolerance=mz_tolerance,
                tolerance_unit=tolerance_unit,
            )
            modified = modified_cosine(
                query_mz,
                query_int,
                reference_mz,
                reference_intensity,
                query.precursor_mz,
                float(self.precursor_mz[index]),
                tolerance=mz_tolerance,
                tolerance_unit=tolerance_unit,
            )
            losses = neutral_loss_cosine(
                query_mz,
                query_int,
                reference_mz,
                reference_intensity,
                query.precursor_mz,
                float(self.precursor_mz[index]),
                tolerance=mz_tolerance,
                tolerance_unit=tolerance_unit,
            )
            reference_mass = float(self.neutral_masses[index])
            mass_delta = query_mass - reference_mass if math.isfinite(query_mass + reference_mass) else None
            ppm = mass_error_ppm(query_mass, reference_mass) if mass_delta is not None else None
            reference_ce_raw = float(self.collision_energies[index])
            reference_ce = reference_ce_raw if math.isfinite(reference_ce_raw) else None
            ce_difference = (
                abs(query_ce - reference_ce)
                if query_ce is not None and reference_ce is not None
                else None
            )
            structure_code = int(self.structure_codes[index])
            hits.append(
                SpectralHit(
                    query_spectrum_id=query.spectrum_id,
                    library_spectrum_id=self.spectrum_ids[index],
                    candidate_smiles=self.structure_smiles[structure_code],
                    candidate_inchikey14=self.structure_keys[structure_code],
                    direct_cosine=direct.score,
                    modified_cosine=modified.score,
                    neutral_loss_cosine=losses.score,
                    matched_peaks=direct.matched_peaks,
                    explained_query_intensity=direct.explained_query_intensity,
                    mass_error_ppm=ppm,
                    mass_delta=mass_delta,
                    matching_adduct=(
                        self.adduct_values[int(self.adduct_codes[index])] == query.adduct
                    ),
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
