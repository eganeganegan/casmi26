from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class Spectrum:
    molecule_id: str
    spectrum_id: str
    precursor_mz: float
    adduct: str
    collision_energy: Any
    mz: np.ndarray
    intensity: np.ndarray
    ionization_mode: str | None = None
    smiles: str | None = None
    canonical_smiles: str | None = None
    inchikey14: str | None = None
    exact_mass: float | None = None
    molecular_formula: str | None = None

    def __post_init__(self) -> None:
        self.mz = np.asarray(self.mz, dtype=np.float64)
        self.intensity = np.asarray(self.intensity, dtype=np.float64)
        if self.mz.ndim != 1 or self.intensity.ndim != 1:
            raise ValueError("Peak arrays must be one-dimensional")
        if len(self.mz) != len(self.intensity):
            raise ValueError("m/z and intensity arrays must have equal lengths")
