from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

IntensityTransform = Literal["raw", "sqrt", "log1p"]


@dataclass(frozen=True, slots=True)
class SpectrumPreprocessingConfig:
    min_relative_intensity: float = 0.01
    top_n: int | None = 256
    remove_precursor_window_da: float | None = 2.0
    intensity_transform: IntensityTransform = "sqrt"
    mz_power: float = 0.0


def parse_peak_array(value: object) -> np.ndarray:
    """Parse Arrow/list/nested singleton/string peak representations."""
    if value is None:
        return np.empty(0, dtype=np.float64)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.empty(0, dtype=np.float64)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                value = np.fromstring(text.strip("[]").replace(",", " "), sep=" ")
    if hasattr(value, "as_py"):
        value = value.as_py()
    array = np.asarray(value, dtype=np.float64)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"Expected one-dimensional peak array, received shape {array.shape}")
    return array


def normalize_spectrum(
    intensities: Iterable[float] | np.ndarray,
    transform: IntensityTransform = "sqrt",
) -> np.ndarray:
    values = np.asarray(intensities, dtype=np.float64)
    if transform == "sqrt":
        values = np.sqrt(np.clip(values, 0.0, None))
    elif transform == "log1p":
        values = np.log1p(np.clip(values, 0.0, None))
    elif transform != "raw":
        raise ValueError(f"Unknown intensity transform: {transform}")
    maximum = values.max(initial=0.0)
    return values / maximum if maximum > 0 else np.zeros_like(values)


def clean_spectrum(
    mz: Iterable[float] | np.ndarray,
    intensity: Iterable[float] | np.ndarray,
    *,
    precursor_mz: float | None = None,
    config: SpectrumPreprocessingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = config or SpectrumPreprocessingConfig()
    mz_arr = parse_peak_array(mz)
    int_arr = parse_peak_array(intensity)
    if mz_arr.shape != int_arr.shape:
        raise ValueError("m/z and intensity arrays must have equal lengths")

    mask = np.isfinite(mz_arr) & np.isfinite(int_arr) & (mz_arr >= 0) & (int_arr >= 0)
    if precursor_mz is not None and cfg.remove_precursor_window_da is not None:
        mask &= mz_arr <= float(precursor_mz) + cfg.remove_precursor_window_da
    mz_arr, int_arr = mz_arr[mask], int_arr[mask]

    if len(int_arr):
        base_peak = int_arr.max(initial=0.0)
        if base_peak > 0 and cfg.min_relative_intensity > 0:
            keep = int_arr >= base_peak * cfg.min_relative_intensity
            mz_arr, int_arr = mz_arr[keep], int_arr[keep]

    if cfg.top_n is not None and len(mz_arr) > cfg.top_n:
        keep = np.argpartition(int_arr, -cfg.top_n)[-cfg.top_n :]
        mz_arr, int_arr = mz_arr[keep], int_arr[keep]

    order = np.argsort(mz_arr, kind="stable")
    mz_arr, int_arr = mz_arr[order], int_arr[order]
    int_arr = normalize_spectrum(int_arr, cfg.intensity_transform)
    if cfg.mz_power:
        int_arr = int_arr * np.power(mz_arr, cfg.mz_power)
        norm = np.linalg.norm(int_arr)
        if norm:
            int_arr /= norm
    return mz_arr, int_arr


def bin_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    max_mz: float = 1200.0,
    bin_width: float = 0.1,
    aggregation: Literal["max", "sum"] = "max",
) -> np.ndarray:
    if max_mz <= 0 or bin_width <= 0:
        raise ValueError("max_mz and bin_width must be positive")
    n_bins = int(np.ceil(max_mz / bin_width))
    output = np.zeros(n_bins, dtype=np.float32)
    indices = np.floor(np.asarray(mz) / bin_width).astype(np.int64)
    valid = (indices >= 0) & (indices < n_bins) & np.isfinite(intensity)
    if aggregation == "max":
        np.maximum.at(output, indices[valid], np.asarray(intensity)[valid])
    elif aggregation == "sum":
        np.add.at(output, indices[valid], np.asarray(intensity)[valid])
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    return output


def neutral_loss_spectrum(
    precursor_mz: float, mz: np.ndarray, intensity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    losses = float(precursor_mz) - np.asarray(mz, dtype=np.float64)
    valid = losses >= 0
    order = np.argsort(losses[valid], kind="stable")
    return losses[valid][order], np.asarray(intensity, dtype=np.float64)[valid][order]
