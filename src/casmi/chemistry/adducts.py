from __future__ import annotations

import re
from dataclasses import dataclass

# 2020 atomic mass evaluation / CODATA masses, in unified atomic mass units.
ELECTRON_MASS = 0.000548579909065
PROTON_MASS = 1.007276466621
HYDROGEN_ATOM_MASS = 1.00782503223
CARBON_12_MASS = 12.0
NITROGEN_14_MASS = 14.00307400443
OXYGEN_16_MASS = 15.99491461957
SODIUM_23_MASS = 22.9897692820
POTASSIUM_39_MASS = 38.9637064864
CHLORINE_35_MASS = 34.968852682
FLUORINE_19_MASS = 18.99840316273
BROMINE_79_MASS = 78.9183376
LITHIUM_7_MASS = 7.0160034366
CALCIUM_40_MASS = 39.962590863
PHOSPHORUS_31_MASS = 30.97376199842
SULFUR_32_MASS = 31.9720711744
WATER_MASS = 2 * HYDROGEN_ATOM_MASS + OXYGEN_16_MASS
FORMIC_ACID_MASS = CARBON_12_MASS + 2 * HYDROGEN_ATOM_MASS + 2 * OXYGEN_16_MASS

_ATOMIC_MASSES = {
    "H": HYDROGEN_ATOM_MASS,
    "C": CARBON_12_MASS,
    "N": NITROGEN_14_MASS,
    "O": OXYGEN_16_MASS,
    "F": FLUORINE_19_MASS,
    "Na": SODIUM_23_MASS,
    "K": POTASSIUM_39_MASS,
    "Cl": CHLORINE_35_MASS,
    "Br": BROMINE_79_MASS,
    "Li": LITHIUM_7_MASS,
    "Ca": CALCIUM_40_MASS,
    "P": PHOSPHORUS_31_MASS,
    "S": SULFUR_32_MASS,
}


@dataclass(frozen=True, slots=True)
class Adduct:
    name: str
    charge: int
    mass_shift: float
    molecule_multiplier: int = 1

    def precursor_mz(self, neutral_mass: float) -> float:
        return (self.molecule_multiplier * neutral_mass + self.mass_shift) / abs(self.charge)

    def neutral_mass(self, precursor_mz: float) -> float:
        return (precursor_mz * abs(self.charge) - self.mass_shift) / self.molecule_multiplier


_ADDUCTS: dict[str, Adduct] = {
    "[M+H]+": Adduct("[M+H]+", 1, PROTON_MASS),
    "[M+NH4]+": Adduct(
        "[M+NH4]+", 1, NITROGEN_14_MASS + 4 * HYDROGEN_ATOM_MASS - ELECTRON_MASS
    ),
    "[M-H2O+H]+": Adduct("[M-H2O+H]+", 1, PROTON_MASS - WATER_MASS),
    "[M-2H2O+H]+": Adduct("[M-2H2O+H]+", 1, PROTON_MASS - 2 * WATER_MASS),
    "[M+Na]+": Adduct("[M+Na]+", 1, SODIUM_23_MASS - ELECTRON_MASS),
    "[M+K]+": Adduct("[M+K]+", 1, POTASSIUM_39_MASS - ELECTRON_MASS),
    "[M-H]-": Adduct("[M-H]-", -1, -PROTON_MASS),
    "[M-H2O-H]-": Adduct("[M-H2O-H]-", -1, -WATER_MASS - PROTON_MASS),
    "[M+CH2O2-H]-": Adduct("[M+CH2O2-H]-", -1, FORMIC_ACID_MASS - PROTON_MASS),
    "[M+Cl]-": Adduct("[M+Cl]-", -1, CHLORINE_35_MASS + ELECTRON_MASS),
    # Frequent training forms.
    "[M]+": Adduct("[M]+", 1, -ELECTRON_MASS),
    "[M]-": Adduct("[M]-", -1, ELECTRON_MASS),
    "[M+2H]2+": Adduct("[M+2H]2+", 2, 2 * PROTON_MASS),
    "[M-2H]2-": Adduct("[M-2H]2-", -2, -2 * PROTON_MASS),
}

_ALIASES = {
    "M+H": "[M+H]+",
    "M-H": "[M-H]-",
    "[M+H]1+": "[M+H]+",
    "[M-H]1-": "[M-H]-",
}

_ADDUCT_PATTERN = re.compile(r"^\[(?P<count>\d*)M(?P<terms>(?:[+-].+)*)\](?P<charge>\d*)(?P<sign>[+-])$")
_TERM_PATTERN = re.compile(r"(?P<sign>[+-])(?P<coefficient>\d*)(?P<formula>[A-Z][A-Za-z0-9]*)")
_ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")


def _formula_mass(formula: str) -> float:
    position = 0
    mass = 0.0
    for match in _ELEMENT_PATTERN.finditer(formula):
        if match.start() != position:
            raise ValueError(f"Unsupported adduct formula: {formula!r}")
        element, raw_count = match.groups()
        if element not in _ATOMIC_MASSES:
            raise ValueError(f"Unsupported adduct element: {element!r}")
        mass += _ATOMIC_MASSES[element] * int(raw_count or 1)
        position = match.end()
    if position != len(formula):
        raise ValueError(f"Unsupported adduct formula: {formula!r}")
    return mass


def _parse_formal_adduct(name: str) -> Adduct:
    match = _ADDUCT_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported adduct: {name!r}")
    molecule_multiplier = int(match.group("count") or 1)
    charge_magnitude = int(match.group("charge") or 1)
    charge = charge_magnitude if match.group("sign") == "+" else -charge_magnitude
    terms = match.group("terms")
    position = 0
    shift = 0.0
    for term in _TERM_PATTERN.finditer(terms):
        if term.start() != position:
            raise ValueError(f"Unsupported adduct terms: {terms!r}")
        coefficient = int(term.group("coefficient") or 1)
        direction = 1.0 if term.group("sign") == "+" else -1.0
        shift += direction * coefficient * _formula_mass(term.group("formula"))
        position = term.end()
    if position != len(terms):
        raise ValueError(f"Unsupported adduct terms: {terms!r}")
    # Neutral atom masses become ionic masses by removing charge * electron mass.
    shift -= charge * ELECTRON_MASS
    return Adduct(name, charge, shift, molecule_multiplier)


def parse_adduct(adduct: str) -> Adduct:
    normalized = adduct.strip().replace(" ", "")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized in _ADDUCTS:
        return _ADDUCTS[normalized]
    return _parse_formal_adduct(normalized)


def neutral_mass_from_precursor(precursor_mz: float, adduct: str) -> float:
    return parse_adduct(adduct).neutral_mass(float(precursor_mz))


def theoretical_precursor_mz(neutral_mass: float, adduct: str) -> float:
    return parse_adduct(adduct).precursor_mz(float(neutral_mass))


def mass_error_ppm(observed: float, expected: float) -> float:
    if expected == 0:
        raise ValueError("Expected mass must be non-zero")
    return 1_000_000.0 * (float(observed) - float(expected)) / float(expected)


def supported_adducts() -> tuple[str, ...]:
    return tuple(_ADDUCTS)
