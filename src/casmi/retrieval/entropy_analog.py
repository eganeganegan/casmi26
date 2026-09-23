from __future__ import annotations

import math
from array import array
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
from numba import njit, prange

from casmi.chemistry import neutral_mass_from_precursor
from casmi.data.schema import detect_columns
from casmi.retrieval.compact_library import (
    CompactSpectralLibraryIndex,
    _PackedStringBuilder,
    _PackedStrings,
)
from casmi.retrieval.library import _neutral_mass
from casmi.spectra import Spectrum, SpectrumPreprocessingConfig, clean_spectrum
from casmi.spectra.preprocessing import parse_peak_array


@dataclass(frozen=True, slots=True)
class EntropyAnalogHit:
    """One structure-level hit from mass-shifted spectral-entropy search."""

    inchikey14: str
    smiles: str
    similarity: float
    neutral_mass: float
    mass_delta: float
    library_spectrum_index: int


@njit(cache=False)
def _select_richest_spectra(
    structure_codes: np.ndarray,
    peak_offsets: np.ndarray,
    structure_count: int,
) -> np.ndarray:
    best = np.full(structure_count, -1, dtype=np.int64)
    best_counts = np.full(structure_count, -1, dtype=np.int64)
    for spectrum_index in range(len(structure_codes)):
        structure_code = int(structure_codes[spectrum_index])
        peak_count = int(peak_offsets[spectrum_index + 1] - peak_offsets[spectrum_index])
        if peak_count > best_counts[structure_code]:
            best[structure_code] = spectrum_index
            best_counts[structure_code] = peak_count
    return best


@njit(cache=False, parallel=True, fastmath=True)
def _prepare_reference_peaks(
    spectrum_indices: np.ndarray,
    source_offsets: np.ndarray,
    source_mz: np.ndarray,
    source_intensity: np.ndarray,
    output_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total_peaks = int(output_offsets[-1])
    output_mz = np.empty(total_peaks, dtype=np.float32)
    output_probability = np.empty(total_peaks, dtype=np.float32)
    for representative_index in prange(len(spectrum_indices)):
        spectrum_index = int(spectrum_indices[representative_index])
        source_start = int(source_offsets[spectrum_index])
        source_end = int(source_offsets[spectrum_index + 1])
        output_start = int(output_offsets[representative_index])
        count = source_end - source_start
        intensity_sum = 0.0
        for offset in range(count):
            intensity_sum += max(float(source_intensity[source_start + offset]), 0.0)
        entropy = 0.0
        for offset in range(count):
            source_position = source_start + offset
            output_position = output_start + offset
            output_mz[output_position] = source_mz[source_position]
            probability = (
                max(float(source_intensity[source_position]), 0.0) / intensity_sum
                if intensity_sum > 0.0
                else 0.0
            )
            output_probability[output_position] = probability
            if probability > 0.0:
                entropy -= probability * math.log(probability)
        if entropy < 3.0 and count:
            exponent = 0.25 + 0.25 * entropy
            weighted_sum = 0.0
            for offset in range(count):
                position = output_start + offset
                weighted = float(output_probability[position]) ** exponent
                output_probability[position] = weighted
                weighted_sum += weighted
            if weighted_sum > 0.0:
                for offset in range(count):
                    position = output_start + offset
                    output_probability[position] /= weighted_sum
    return output_mz, output_probability


@njit(cache=False, fastmath=True)
def _entropy_similarity(
    query_mz: np.ndarray,
    query_probability: np.ndarray,
    reference_mz: np.ndarray,
    reference_probability: np.ndarray,
    tolerance_da: float,
    reference_shift_da: float,
) -> float:
    """Jensen-Shannon spectral similarity with greedy sorted-peak matching."""
    query_entropy = 0.0
    reference_entropy = 0.0
    for probability in query_probability:
        if probability > 0.0:
            query_entropy -= probability * math.log(probability)
    for probability in reference_probability:
        if probability > 0.0:
            reference_entropy -= probability * math.log(probability)

    query_index = 0
    reference_index = 0
    mixture_entropy = 0.0
    while query_index < len(query_mz) and reference_index < len(reference_mz):
        delta = float(query_mz[query_index]) - (
            float(reference_mz[reference_index]) + reference_shift_da
        )
        if delta < -tolerance_da:
            probability = 0.5 * float(query_probability[query_index])
            query_index += 1
        elif delta > tolerance_da:
            probability = 0.5 * float(reference_probability[reference_index])
            reference_index += 1
        else:
            probability = 0.5 * (
                float(query_probability[query_index])
                + float(reference_probability[reference_index])
            )
            query_index += 1
            reference_index += 1
        if probability > 0.0:
            mixture_entropy -= probability * math.log(probability)
    while query_index < len(query_mz):
        probability = 0.5 * float(query_probability[query_index])
        if probability > 0.0:
            mixture_entropy -= probability * math.log(probability)
        query_index += 1
    while reference_index < len(reference_mz):
        probability = 0.5 * float(reference_probability[reference_index])
        if probability > 0.0:
            mixture_entropy -= probability * math.log(probability)
        reference_index += 1

    divergence = mixture_entropy - 0.5 * (query_entropy + reference_entropy)
    similarity = 1.0 - divergence / math.log(2.0)
    return min(1.0, max(0.0, similarity))


@njit(cache=False, parallel=True, fastmath=True)
def _search_representatives(
    query_mz: np.ndarray,
    query_probability: np.ndarray,
    candidate_indices: np.ndarray,
    peak_offsets: np.ndarray,
    peak_mz: np.ndarray,
    peak_probability: np.ndarray,
    mass_shifts: np.ndarray,
    tolerance_da: float,
) -> np.ndarray:
    scores = np.zeros(len(candidate_indices), dtype=np.float32)
    for candidate_position in prange(len(candidate_indices)):
        representative_index = int(candidate_indices[candidate_position])
        start = int(peak_offsets[representative_index])
        end = int(peak_offsets[representative_index + 1])
        if start == end:
            continue
        direct = _entropy_similarity(
            query_mz,
            query_probability,
            peak_mz[start:end],
            peak_probability[start:end],
            tolerance_da,
            0.0,
        )
        shift = float(mass_shifts[candidate_position])
        if abs(shift) > 0.001:
            shifted = _entropy_similarity(
                query_mz,
                query_probability,
                peak_mz[start:end],
                peak_probability[start:end],
                tolerance_da,
                shift,
            )
            if shifted > direct:
                direct = shifted
        scores[candidate_position] = direct
    return scores


@njit(cache=False)
def _best_scores_by_structure(
    candidate_indices: np.ndarray,
    structure_codes: np.ndarray,
    representative_scores: np.ndarray,
    allowed: np.ndarray,
    structure_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse representative scores to the best representative per structure."""
    best_scores = np.zeros(structure_count, dtype=np.float32)
    best_positions = np.full(structure_count, -1, dtype=np.int64)
    for candidate_position in range(len(candidate_indices)):
        if not allowed[candidate_position]:
            continue
        representative_index = int(candidate_indices[candidate_position])
        structure_code = int(structure_codes[representative_index])
        score = representative_scores[candidate_position]
        if score > best_scores[structure_code]:
            best_scores[structure_code] = score
            best_positions[structure_code] = candidate_position
    return best_scores, best_positions


def _entropy_weight(intensity: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(intensity, dtype=np.float32), 0.0, None)
    total = float(probability.sum())
    if total <= 0.0:
        return np.zeros_like(probability)
    probability /= total
    positive = probability > 0
    entropy = float(-(probability[positive] * np.log(probability[positive])).sum())
    if entropy < 3.0:
        probability = np.power(probability, 0.25 + 0.25 * entropy)
        probability /= float(probability.sum())
    return probability.astype(np.float32, copy=False)


class RepresentativeEntropyIndex:
    """One rich spectrum per structure for exhaustive mass-shifted analog search.

    The underlying compact library has already removed noise, capped peak counts, and
    sorted peaks. This index entropy-weights those retained intensities once and keeps
    only the richest spectrum for each structure.
    """

    def __init__(self, library: CompactSpectralLibraryIndex) -> None:
        self.library = library
        self.preprocessing = library.preprocessing
        selected = _select_richest_spectra(
            library.structure_codes,
            library.peak_offsets,
            len(library.structure_keys),
        )
        valid = (selected >= 0) & np.isfinite(library.neutral_masses[np.maximum(selected, 0)])
        selected = selected[valid]
        order = np.argsort(library.neutral_masses[selected], kind="stable")
        self.spectrum_indices = selected[order].astype(np.int64, copy=False)
        self.neutral_masses = library.neutral_masses[self.spectrum_indices].astype(
            np.float64, copy=True
        )
        self.structure_codes = library.structure_codes[self.spectrum_indices].astype(
            np.uint32, copy=True
        )
        self.polarities = library.polarities[self.spectrum_indices].astype(np.int8, copy=True)
        peak_counts = (
            library.peak_offsets[self.spectrum_indices + 1]
            - library.peak_offsets[self.spectrum_indices]
        ).astype(np.uint64)
        self.peak_offsets = np.empty(len(self.spectrum_indices) + 1, dtype=np.uint64)
        self.peak_offsets[0] = 0
        np.cumsum(peak_counts, out=self.peak_offsets[1:])
        self.peak_mz, self.peak_probability = _prepare_reference_peaks(
            self.spectrum_indices,
            library.peak_offsets,
            library.peak_mz,
            library.peak_intensity,
            self.peak_offsets,
        )
        self._structure_code_by_key: dict[str, int] | None = None

    @property
    def structure_count(self) -> int:
        return len(self.library.structure_keys)

    def _structure_key(self, code: int) -> str:
        return self.library.structure_keys[code]

    def _structure_smiles(self, code: int) -> str:
        return self.library.structure_smiles[code]

    def __len__(self) -> int:
        return len(self.spectrum_indices)

    @property
    def peak_count(self) -> int:
        return len(self.peak_mz)

    @property
    def storage_nbytes(self) -> int:
        arrays = (
            self.spectrum_indices,
            self.neutral_masses,
            self.structure_codes,
            self.polarities,
            self.peak_offsets,
            self.peak_mz,
            self.peak_probability,
        )
        return sum(array.nbytes for array in arrays)

    def search_molecule(
        self,
        spectra: Iterable[Spectrum],
        *,
        top_n: int = 100,
        mass_window_da: float = 200.0,
        mz_tolerance_da: float = 0.02,
        exclude_inchikey14: Collection[str] | None = None,
        require_same_polarity: bool = False,
        inputs_preprocessed: bool = False,
    ) -> list[EntropyAnalogHit]:
        spectra_list = list(spectra)
        if not spectra_list or top_n <= 0:
            return []
        masses = np.asarray([_neutral_mass(spectrum) for spectrum in spectra_list])
        finite_masses = masses[np.isfinite(masses)]
        if not len(finite_masses):
            return []
        target_mass = float(np.median(finite_masses))
        left = int(np.searchsorted(self.neutral_masses, target_mass - mass_window_da, "left"))
        right = int(np.searchsorted(self.neutral_masses, target_mass + mass_window_da, "right"))
        candidates = np.arange(left, right, dtype=np.int64)
        if not len(candidates):
            return []

        excluded_keys = set(exclude_inchikey14 or ())
        plausible_keys = {
            key
            for key in excluded_keys
            if len(key) == 14 and key.isalnum() and key.upper() == key
        }
        excluded_codes: set[int] = set()
        if plausible_keys:
            if getattr(self, "_structure_code_by_key", None) is None:
                self._structure_code_by_key = {
                    self._structure_key(code): code
                    for code in range(self.structure_count)
                }
            excluded_codes = {
                self._structure_code_by_key[key]
                for key in plausible_keys
                if key in self._structure_code_by_key
            }
        allowed = np.ones(len(candidates), dtype=bool)
        if excluded_codes:
            allowed &= ~np.isin(self.structure_codes[candidates], list(excluded_codes))
        mass_shifts = (target_mass - self.neutral_masses[candidates]).astype(np.float32)
        aggregate = np.zeros(len(candidates), dtype=np.float32)

        for spectrum in spectra_list:
            if inputs_preprocessed:
                query_mz = np.asarray(spectrum.mz, dtype=np.float32)
                query_intensity = np.asarray(spectrum.intensity, dtype=np.float32)
            else:
                query_mz, query_intensity = clean_spectrum(
                    spectrum.mz,
                    spectrum.intensity,
                    precursor_mz=spectrum.precursor_mz,
                    config=self.preprocessing,
                )
                query_mz = query_mz.astype(np.float32, copy=False)
                query_intensity = query_intensity.astype(np.float32, copy=False)
            if not len(query_mz):
                continue
            query_probability = _entropy_weight(query_intensity)
            spectrum_allowed = allowed.copy()
            if require_same_polarity:
                polarity = (
                    1
                    if (spectrum.ionization_mode or "").lower() == "positive"
                    else -1
                    if (spectrum.ionization_mode or "").lower() == "negative"
                    else 0
                )
                if polarity:
                    spectrum_allowed &= self.polarities[candidates] == polarity
            local_positions = np.flatnonzero(spectrum_allowed)
            if not len(local_positions):
                continue
            local_scores = _search_representatives(
                query_mz,
                query_probability,
                candidates[local_positions],
                self.peak_offsets,
                self.peak_mz,
                self.peak_probability,
                mass_shifts[local_positions],
                mz_tolerance_da,
            )
            aggregate[local_positions] = np.maximum(aggregate[local_positions], local_scores)

        structure_scores, best_positions = _best_scores_by_structure(
            candidates,
            self.structure_codes,
            aggregate,
            allowed,
            self.structure_count,
        )
        eligible_codes = np.flatnonzero(best_positions >= 0)
        if not len(eligible_codes):
            return []
        keep = min(top_n, len(eligible_codes))
        if keep < len(eligible_codes):
            local = np.argpartition(structure_scores[eligible_codes], -keep)[-keep:]
            eligible_codes = eligible_codes[local]
        eligible_codes = eligible_codes[
            np.argsort(structure_scores[eligible_codes], kind="stable")[::-1]
        ]
        hits: list[EntropyAnalogHit] = []
        for structure_code in eligible_codes:
            position = int(best_positions[structure_code])
            representative_index = int(candidates[position])
            reference_mass = float(self.neutral_masses[representative_index])
            hits.append(
                EntropyAnalogHit(
                    inchikey14=self._structure_key(structure_code),
                    smiles=self._structure_smiles(structure_code),
                    similarity=float(aggregate[position]),
                    neutral_mass=reference_mass,
                    mass_delta=target_mass - reference_mass,
                    library_spectrum_index=int(self.spectrum_indices[representative_index]),
                )
            )
        return hits


class RawRepresentativeEntropyIndex(RepresentativeEntropyIndex):
    """Standalone entropy index built from raw, linear-intensity spectra.

    One richest spectrum is retained per structure before applying the entropy-search
    preprocessing. Unlike :class:`RepresentativeEntropyIndex`, this artifact does not
    embed the multi-million-spectrum compact cosine library.
    """

    DEFAULT_PREPROCESSING = SpectrumPreprocessingConfig(
        min_relative_intensity=0.002,
        top_n=256,
        remove_precursor_window_da=None,
        intensity_transform="raw",
    )

    def __init__(
        self,
        *,
        spectrum_indices: np.ndarray,
        neutral_masses: np.ndarray,
        polarities: np.ndarray,
        peak_offsets: np.ndarray,
        peak_mz: np.ndarray,
        peak_probability: np.ndarray,
        structure_keys: _PackedStrings,
        structure_smiles: _PackedStrings,
        structure_codes: np.ndarray | None = None,
        preprocessing: SpectrumPreprocessingConfig | None = None,
    ) -> None:
        self.library = None
        self.preprocessing = preprocessing or self.DEFAULT_PREPROCESSING
        self.spectrum_indices = np.asarray(spectrum_indices, dtype=np.int64)
        self.neutral_masses = np.asarray(neutral_masses, dtype=np.float64)
        self.polarities = np.asarray(polarities, dtype=np.int8)
        self.peak_offsets = np.asarray(peak_offsets, dtype=np.uint64)
        self.peak_mz = np.asarray(peak_mz, dtype=np.float32)
        self.peak_probability = np.asarray(peak_probability, dtype=np.float32)
        self.structure_keys = structure_keys
        self.structure_smiles = structure_smiles
        self.structure_codes = (
            np.arange(len(self.neutral_masses), dtype=np.uint32)
            if structure_codes is None
            else np.asarray(structure_codes, dtype=np.uint32)
        )
        self._structure_code_by_key: dict[str, int] | None = None
        count = len(self.neutral_masses)
        if not (
            len(self.spectrum_indices)
            == len(self.polarities)
            == len(self.structure_codes)
            == count
        ):
            raise ValueError("Raw representative metadata arrays must have equal lengths")
        if len(self.structure_keys) != len(self.structure_smiles):
            raise ValueError("Raw representative structure metadata must have equal lengths")
        if count and int(self.structure_codes.max()) >= len(self.structure_keys):
            raise ValueError("Raw representative structure code is out of range")
        if len(self.peak_offsets) != count + 1 or int(self.peak_offsets[-1]) != len(self.peak_mz):
            raise ValueError("Raw representative peak offsets are inconsistent")
        if len(self.peak_probability) != len(self.peak_mz):
            raise ValueError("Raw representative peak arrays must have equal lengths")
        if count and np.any(np.diff(self.neutral_masses) < 0):
            raise ValueError("Raw representative neutral masses must be sorted")

    def _structure_key(self, code: int) -> str:
        return self.structure_keys[code]

    def _structure_smiles(self, code: int) -> str:
        return self.structure_smiles[code]

    @property
    def structure_count(self) -> int:
        return len(self.structure_keys)

    @property
    def storage_nbytes(self) -> int:
        arrays = (
            self.spectrum_indices,
            self.neutral_masses,
            self.structure_codes,
            self.polarities,
            self.peak_offsets,
            self.peak_mz,
            self.peak_probability,
            self.structure_keys.data,
            self.structure_keys.offsets,
            self.structure_smiles.data,
            self.structure_smiles.offsets,
        )
        return sum(value.nbytes for value in arrays)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> RawRepresentativeEntropyIndex:
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(value).__name__}")
        return value

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        batch_size: int = 16_384,
        per_polarity: bool = False,
        preprocessing: SpectrumPreprocessingConfig | None = None,
    ) -> RawRepresentativeEntropyIndex:
        """Build the standalone index in two streaming passes over a parquet file."""
        cfg = preprocessing or cls.DEFAULT_PREPROCESSING
        parquet = pq.ParquetFile(path)
        cmap = detect_columns(parquet.schema_arrow.names)
        if not cmap.inchikey14 or not cmap.smiles or not cmap.adduct or not cmap.ionization_mode:
            raise ValueError("Raw entropy indexing requires structure, adduct, and polarity columns")
        count_column = "num_peaks" if "num_peaks" in parquet.schema_arrow.names else None
        metadata_columns = [
            cmap.inchikey14,
            cmap.smiles,
            cmap.precursor_mz,
            cmap.adduct,
            cmap.ionization_mode,
        ]
        if count_column:
            metadata_columns.append(count_column)
        else:
            metadata_columns.append(cmap.mz)

        # selection key -> (structure key, peak count, global row, smiles, mass, polarity)
        richest: dict[tuple[str, int], tuple[str, int, int, str, float, int]] = {}
        row_offset = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=metadata_columns):
            values = batch.to_pydict()
            for local_index in range(batch.num_rows):
                key_value = values[cmap.inchikey14][local_index]
                smiles_value = values[cmap.smiles][local_index]
                if key_value is None or smiles_value is None:
                    continue
                peak_count = (
                    int(values[count_column][local_index] or 0)
                    if count_column
                    else len(parse_peak_array(values[cmap.mz][local_index]))
                )
                key = str(key_value)
                try:
                    neutral_mass = neutral_mass_from_precursor(
                        float(values[cmap.precursor_mz][local_index]),
                        str(values[cmap.adduct][local_index] or ""),
                    )
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
                if not np.isfinite(neutral_mass):
                    continue
                mode = str(values[cmap.ionization_mode][local_index] or "").lower()
                polarity = 1 if mode == "positive" else -1 if mode == "negative" else 0
                selection_key = (key, polarity if per_polarity else 0)
                previous = richest.get(selection_key)
                if previous is not None and peak_count <= previous[1]:
                    continue
                richest[selection_key] = (
                    key,
                    peak_count,
                    row_offset + local_index,
                    str(smiles_value),
                    neutral_mass,
                    polarity,
                )
            row_offset += batch.num_rows

        selected = {metadata[2]: metadata for metadata in richest.values()}
        source_mz = array("f")
        source_intensity = array("f")
        source_offsets = array("Q", [0])
        source_rows: list[int] = []
        keys: list[str] = []
        smiles: list[str] = []
        masses: list[float] = []
        polarities: list[int] = []
        selected_rows = sorted(selected)
        selected_position = 0
        row_offset = 0
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=[cmap.mz, cmap.intensity],
        ):
            values = batch.to_pydict()
            batch_end = row_offset + batch.num_rows
            while (
                selected_position < len(selected_rows)
                and selected_rows[selected_position] < batch_end
            ):
                global_row = selected_rows[selected_position]
                selected_position += 1
                local_index = global_row - row_offset
                metadata = selected[global_row]
                mz, intensity = clean_spectrum(
                    values[cmap.mz][local_index],
                    values[cmap.intensity][local_index],
                    config=cfg,
                )
                if not len(mz):
                    continue
                source_mz.frombytes(np.asarray(mz, dtype=np.float32).tobytes())
                source_intensity.frombytes(np.asarray(intensity, dtype=np.float32).tobytes())
                source_offsets.append(len(source_mz))
                source_rows.append(global_row)
                keys.append(metadata[0])
                smiles.append(metadata[3])
                masses.append(metadata[4])
                polarities.append(metadata[5])
            row_offset = batch_end

        mass_values = np.asarray(masses, dtype=np.float64)
        order = np.argsort(mass_values, kind="stable").astype(np.int64)
        raw_offsets = np.frombuffer(source_offsets, dtype=np.uint64).copy()
        counts = raw_offsets[order + 1] - raw_offsets[order]
        output_offsets = np.empty(len(order) + 1, dtype=np.uint64)
        output_offsets[0] = 0
        np.cumsum(counts, out=output_offsets[1:])
        peak_mz, peak_probability = _prepare_reference_peaks(
            order,
            raw_offsets,
            np.frombuffer(source_mz, dtype=np.float32),
            np.frombuffer(source_intensity, dtype=np.float32),
            output_offsets,
        )
        key_builder = _PackedStringBuilder()
        smiles_builder = _PackedStringBuilder()
        key_to_code: dict[str, int] = {}
        structure_codes: list[int] = []
        for source_index in order:
            source_position = int(source_index)
            key = keys[source_position]
            structure_code = key_to_code.get(key)
            if structure_code is None:
                structure_code = len(key_to_code)
                key_to_code[key] = structure_code
                key_builder.append(key)
                smiles_builder.append(smiles[source_position])
            structure_codes.append(structure_code)
        return cls(
            spectrum_indices=np.asarray(source_rows, dtype=np.int64)[order],
            neutral_masses=mass_values[order],
            polarities=np.asarray(polarities, dtype=np.int8)[order],
            peak_offsets=output_offsets,
            peak_mz=peak_mz,
            peak_probability=peak_probability,
            structure_keys=key_builder.finish(),
            structure_smiles=smiles_builder.finish(),
            structure_codes=np.asarray(structure_codes, dtype=np.uint32),
            preprocessing=cfg,
        )
