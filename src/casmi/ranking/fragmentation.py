from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from casmi.chemistry import parse_adduct
from casmi.spectra import Spectrum

PROTON_MASS = 1.007276466621
HYDROGEN_ATOM_MASS = 1.00782503223
ELECTRON_MASS = 0.000548579909065
COMMON_NEUTRAL_LOSSES = np.asarray(
    [
        17.026549101,  # NH3
        18.010564684,  # H2O
        27.994914620,  # CO
        30.010564684,  # CH2O
        43.989829240,  # CO2
        46.005479304,  # formic acid
    ],
    dtype=np.float64,
)

FRAGMENT_FEATURES = (
    "fragment_scored",
    "fragment_theoretical_count",
    "fragment_peak_fraction_max",
    "fragment_peak_fraction_mean",
    "fragment_intensity_fraction_max",
    "fragment_intensity_fraction_mean",
    "fragment_matched_count_max",
    "fragment_matched_count_mean",
    "fragment_top_intensity_fraction_max",
    "fragment_top_intensity_fraction_mean",
    "fragment_mass_weighted_score_max",
    "fragment_mass_weighted_score_mean",
    "fragment_neutral_loss_score_max",
    "fragment_neutral_loss_score_mean",
)


@dataclass(frozen=True, slots=True)
class FragmentationConfig:
    mz_tolerance_da: float = 0.02
    max_bonds: int = 32
    max_spectra: int = 8
    top_intensity_peaks: int = 10
    min_fragment_mass: float = 15.0

    def __post_init__(self) -> None:
        if self.mz_tolerance_da <= 0:
            raise ValueError("mz_tolerance_da must be positive")
        if self.max_bonds <= 0 or self.max_spectra <= 0:
            raise ValueError("max_bonds and max_spectra must be positive")
        if self.top_intensity_peaks <= 0:
            raise ValueError("top_intensity_peaks must be positive")


@lru_cache(maxsize=250_000)
def _theoretical_fragment_masses_cached(
    smiles: str,
    max_bonds: int,
    min_fragment_mass: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    Chem.SanitizeMol(mol)
    parent_mass = float(rdMolDescriptors.CalcExactMolWt(mol))
    eligible = [
        bond
        for bond in mol.GetBonds()
        if not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE
    ][:max_bonds]
    fragment_masses: set[float] = set()
    for bond in eligible:
        broken = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=False)
        try:
            fragments = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
        except Chem.rdchem.MolSanitizeException:
            continue
        for fragment in fragments:
            mass = float(rdMolDescriptors.CalcExactMolWt(fragment))
            if min_fragment_mass <= mass < parent_mass - 0.5:
                fragment_masses.add(round(mass, 6))
    # Common intact neutral losses remain useful when no eligible bond is identified.
    for loss in COMMON_NEUTRAL_LOSSES:
        product_mass = parent_mass - float(loss)
        if product_mass >= min_fragment_mass:
            fragment_masses.add(round(product_mass, 6))
    masses = np.asarray(sorted(fragment_masses), dtype=np.float64)
    losses = parent_mass - masses
    losses = losses[losses > 0]
    all_losses = np.unique(np.concatenate((losses, COMMON_NEUTRAL_LOSSES)))
    return tuple(masses.tolist()), tuple(all_losses.tolist())


def theoretical_fragment_masses(
    smiles: str,
    *,
    max_bonds: int = 32,
    min_fragment_mass: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cached neutral product and neutral-loss masses for a candidate."""
    products, losses = _theoretical_fragment_masses_cached(
        smiles, int(max_bonds), float(min_fragment_mass)
    )
    return np.asarray(products, dtype=np.float64), np.asarray(losses, dtype=np.float64)


def _polarity(spectrum: Spectrum) -> int:
    mode = (spectrum.ionization_mode or "").lower()
    if "pos" in mode or mode == "+":
        return 1
    if "neg" in mode or mode == "-":
        return -1
    try:
        return 1 if parse_adduct(spectrum.adduct).charge > 0 else -1
    except ValueError:
        return 0


def _product_ion_masses(neutral_masses: np.ndarray, polarity: int) -> np.ndarray:
    if not len(neutral_masses):
        return neutral_masses
    if polarity > 0:
        shifts = np.asarray(
            [-HYDROGEN_ATOM_MASS - ELECTRON_MASS, -PROTON_MASS, 0.0, PROTON_MASS]
        )
    elif polarity < 0:
        shifts = np.asarray(
            [-PROTON_MASS, 0.0, HYDROGEN_ATOM_MASS + ELECTRON_MASS]
        )
    else:
        shifts = np.asarray([-PROTON_MASS, 0.0, PROTON_MASS])
    values = (neutral_masses[:, None] + shifts[None, :]).ravel()
    return np.unique(values[values > 0])


def _within_tolerance(
    observed: np.ndarray, theoretical: np.ndarray, tolerance: float
) -> np.ndarray:
    if not len(observed) or not len(theoretical):
        return np.zeros(len(observed), dtype=bool)
    expected = np.sort(theoretical)
    indices = np.searchsorted(expected, observed)
    right = np.minimum(indices, len(expected) - 1)
    left = np.maximum(indices - 1, 0)
    distance = np.minimum(np.abs(observed - expected[left]), np.abs(observed - expected[right]))
    return distance <= tolerance


def _score_spectrum(
    spectrum: Spectrum,
    product_masses: np.ndarray,
    neutral_losses: np.ndarray,
    config: FragmentationConfig,
) -> np.ndarray:
    mz = np.asarray(spectrum.mz, dtype=np.float64)
    intensity = np.asarray(spectrum.intensity, dtype=np.float64)
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity >= 0)
    mz, intensity = mz[valid], intensity[valid]
    if not len(mz):
        return np.zeros(6, dtype=np.float64)
    ions = _product_ion_masses(product_masses, _polarity(spectrum))
    matched = _within_tolerance(mz, ions, config.mz_tolerance_da)
    total_intensity = float(intensity.sum())
    weighted = intensity * mz
    total_weighted = float(weighted.sum())
    top_count = min(config.top_intensity_peaks, len(mz))
    top_indices = np.argpartition(intensity, -top_count)[-top_count:]
    losses = float(spectrum.precursor_mz) - mz
    valid_losses = losses > 0
    loss_matched = _within_tolerance(
        losses[valid_losses], neutral_losses, config.mz_tolerance_da
    )
    loss_intensity = intensity[valid_losses]
    return np.asarray(
        [
            float(matched.mean()),
            float(intensity[matched].sum() / total_intensity) if total_intensity else 0.0,
            float(matched.sum()),
            float(matched[top_indices].mean()),
            float(weighted[matched].sum() / total_weighted) if total_weighted else 0.0,
            (
                float(loss_intensity[loss_matched].sum() / loss_intensity.sum())
                if loss_intensity.sum()
                else 0.0
            ),
        ],
        dtype=np.float64,
    )


def score_candidate_fragments(
    smiles: str,
    spectra: Iterable[Spectrum],
    config: FragmentationConfig | None = None,
) -> dict[str, float]:
    """Score bounded, heuristic in-silico fragments against a molecule's observed spectra."""
    cfg = config or FragmentationConfig()
    products, losses = theoretical_fragment_masses(
        smiles,
        max_bonds=cfg.max_bonds,
        min_fragment_mass=cfg.min_fragment_mass,
    )
    selected_spectra = list(spectra)[: cfg.max_spectra]
    if not selected_spectra or not len(products):
        return {feature: 0.0 for feature in FRAGMENT_FEATURES}
    values = np.stack(
        [_score_spectrum(spectrum, products, losses, cfg) for spectrum in selected_spectra]
    )
    result: dict[str, float] = {
        "fragment_scored": 1.0,
        "fragment_theoretical_count": float(len(products)),
    }
    names = (
        "peak_fraction",
        "intensity_fraction",
        "matched_count",
        "top_intensity_fraction",
        "mass_weighted_score",
        "neutral_loss_score",
    )
    for index, name in enumerate(names):
        result[f"fragment_{name}_max"] = float(values[:, index].max())
        result[f"fragment_{name}_mean"] = float(values[:, index].mean())
    return result
