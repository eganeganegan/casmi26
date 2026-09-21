from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from casmi.chemistry import neutral_mass_from_precursor
from casmi.spectra import Spectrum, SpectrumPreprocessingConfig, bin_spectrum, clean_spectrum

try:
    import torch
    from torch import nn
except ImportError:  # The retrieval-only baseline intentionally does not require PyTorch.
    torch = None
    nn = None


@dataclass(frozen=True, slots=True)
class FingerprintFeatureConfig:
    max_mz: float = 1200.0
    bin_width: float = 1.0
    bin_aggregation: Literal["max", "sum"] = "max"
    include_metadata: bool = True
    mass_scale: float = 1000.0
    collision_energy_scale: float = 100.0

    @property
    def spectrum_bins(self) -> int:
        return int(math.ceil(self.max_mz / self.bin_width))

    @property
    def metadata_features(self) -> int:
        # Neutral mass, precursor m/z, collision energy, and positive/negative polarity.
        return 5 if self.include_metadata else 0

    @property
    def input_dim(self) -> int:
        return self.spectrum_bins + self.metadata_features


@dataclass(frozen=True, slots=True)
class FingerprintModelConfig:
    n_bits: int = 2048
    hidden_dims: tuple[int, ...] = (1024, 512)
    dropout: float = 0.15


def _collision_energy(value: object) -> float:
    if value is None:
        return 0.0
    try:
        array = np.asarray(value, dtype=np.float64)
        finite = array[np.isfinite(array)]
        return float(finite.mean()) if finite.size else 0.0
    except (TypeError, ValueError):
        return 0.0


def spectrum_fingerprint_features(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    precursor_mz: float,
    adduct: str,
    collision_energy: object = None,
    ionization_mode: str | None = None,
    config: FingerprintFeatureConfig | None = None,
) -> np.ndarray:
    """Encode an already-cleaned spectrum and observed metadata as a dense vector."""
    cfg = config or FingerprintFeatureConfig()
    binned = bin_spectrum(
        np.asarray(mz),
        np.asarray(intensity),
        max_mz=cfg.max_mz,
        bin_width=cfg.bin_width,
        aggregation=cfg.bin_aggregation,
    )
    maximum = float(binned.max(initial=0.0))
    if maximum > 0:
        binned /= maximum
    if not cfg.include_metadata:
        return binned
    try:
        neutral_mass = neutral_mass_from_precursor(precursor_mz, adduct)
    except ValueError:
        neutral_mass = 0.0
    polarity = (ionization_mode or "").lower()
    positive = float(polarity.startswith("pos") or (not polarity and adduct.endswith("+")))
    negative = float(polarity.startswith("neg") or (not polarity and adduct.endswith("-")))
    metadata = np.asarray(
        [
            neutral_mass / cfg.mass_scale,
            float(precursor_mz) / cfg.mass_scale,
            _collision_energy(collision_energy) / cfg.collision_energy_scale,
            positive,
            negative,
        ],
        dtype=np.float32,
    )
    metadata[~np.isfinite(metadata)] = 0.0
    return np.concatenate((binned, metadata))


def featurize_spectrum(
    spectrum: Spectrum,
    *,
    feature_config: FingerprintFeatureConfig | None = None,
    preprocessing: SpectrumPreprocessingConfig | None = None,
) -> np.ndarray:
    mz, intensity = clean_spectrum(
        spectrum.mz,
        spectrum.intensity,
        precursor_mz=spectrum.precursor_mz,
        config=preprocessing,
    )
    return spectrum_fingerprint_features(
        mz,
        intensity,
        precursor_mz=spectrum.precursor_mz,
        adduct=spectrum.adduct,
        collision_energy=spectrum.collision_energy,
        ionization_mode=spectrum.ionization_mode,
        config=feature_config,
    )


def pool_fingerprint_probabilities(
    probabilities: np.ndarray,
    method: Literal["mean", "confidence_weighted"] = "mean",
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("probabilities must be a non-empty [spectra, bits] array")
    if method == "mean":
        return values.mean(axis=0).astype(np.float32)
    if method == "confidence_weighted":
        weights = np.mean(np.abs(values - 0.5) * 2.0, axis=1)
        if not np.any(weights):
            return values.mean(axis=0).astype(np.float32)
        return np.average(values, axis=0, weights=weights).astype(np.float32)
    raise ValueError(f"Unknown fingerprint pooling method: {method}")


def fingerprint_compatibility(
    probabilities: np.ndarray,
    candidate_fingerprints: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorized soft/binary similarity features for candidate ranking."""
    probs = np.asarray(probabilities, dtype=np.float64)
    candidates = np.asarray(candidate_fingerprints, dtype=np.float64)
    if probs.ndim != 1 or candidates.ndim != 2 or candidates.shape[1] != probs.shape[0]:
        raise ValueError("Expected probabilities [bits] and candidate_fingerprints [candidates, bits]")
    intersection = candidates @ probs
    candidate_count = candidates.sum(axis=1)
    probability_count = probs.sum()
    union = candidate_count + probability_count - intersection
    expected_tanimoto = np.divide(
        intersection,
        union,
        out=np.ones_like(intersection),
        where=union > 0,
    )
    binary = probs >= 0.5
    binary_intersection = np.count_nonzero(candidates.astype(bool) & binary, axis=1)
    binary_union = np.count_nonzero(candidates.astype(bool) | binary, axis=1)
    binary_tanimoto = np.divide(
        binary_intersection,
        binary_union,
        out=np.ones_like(binary_intersection, dtype=np.float64),
        where=binary_union > 0,
    )
    norms = np.linalg.norm(candidates, axis=1) * np.linalg.norm(probs)
    cosine = np.divide(
        intersection,
        norms,
        out=np.zeros_like(intersection),
        where=norms > 0,
    )
    eps = 1e-7
    clipped = np.clip(probs, eps, 1.0 - eps)
    bce = -np.mean(
        candidates * np.log(clipped) + (1.0 - candidates) * np.log(1.0 - clipped),
        axis=1,
    )
    return {
        "soft_intersection": intersection.astype(np.float32),
        "expected_tanimoto": expected_tanimoto.astype(np.float32),
        "binary_tanimoto": binary_tanimoto.astype(np.float32),
        "fingerprint_cosine": cosine.astype(np.float32),
        "fingerprint_bce": bce.astype(np.float32),
    }


if nn is not None:

    class SpectrumFingerprintMLP(nn.Module):
        def __init__(
            self,
            input_dim: int,
            config: FingerprintModelConfig | None = None,
        ) -> None:
            super().__init__()
            cfg = config or FingerprintModelConfig()
            layers: list[nn.Module] = []
            previous = input_dim
            for width in cfg.hidden_dims:
                layers.extend(
                    [
                        nn.Linear(previous, width),
                        nn.LayerNorm(width),
                        nn.GELU(),
                        nn.Dropout(cfg.dropout),
                    ]
                )
                previous = width
            layers.append(nn.Linear(previous, cfg.n_bits))
            self.network = nn.Sequential(*layers)
            self.model_config = cfg
            self.input_dim = input_dim

        def forward(self, features):  # type: ignore[no-untyped-def]
            return self.network(features)

else:

    class SpectrumFingerprintMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            raise ImportError("PyTorch is required; install the project with `pip install -e '.[ml]'`")


def require_torch():  # type: ignore[no-untyped-def]
    if torch is None:
        raise ImportError("PyTorch is required; install the project with `pip install -e '.[ml]'`")
    return torch


def save_fingerprint_model(
    model: SpectrumFingerprintMLP,
    path: str | Path,
    *,
    feature_config: FingerprintFeatureConfig,
    model_config: FingerprintModelConfig,
    extra: dict[str, object] | None = None,
) -> None:
    torch_module = require_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(
        {
            "state_dict": model.state_dict(),
            "feature_config": asdict(feature_config),
            "model_config": asdict(model_config),
            "extra": extra or {},
        },
        destination,
    )


def load_fingerprint_model(
    path: str | Path,
    *,
    map_location: str = "cpu",
) -> tuple[SpectrumFingerprintMLP, FingerprintFeatureConfig, FingerprintModelConfig, dict[str, object]]:
    torch_module = require_torch()
    bundle = torch_module.load(path, map_location=map_location, weights_only=True)
    feature_config = FingerprintFeatureConfig(**bundle["feature_config"])
    model_values = dict(bundle["model_config"])
    model_values["hidden_dims"] = tuple(model_values["hidden_dims"])
    model_config = FingerprintModelConfig(**model_values)
    model = SpectrumFingerprintMLP(feature_config.input_dim, model_config)
    model.load_state_dict(bundle["state_dict"])
    return model, feature_config, model_config, dict(bundle.get("extra", {}))
