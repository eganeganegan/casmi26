import pytest

from casmi.chemistry.adducts import (
    WATER_MASS,
    neutral_mass_from_precursor,
    supported_adducts,
    theoretical_precursor_mz,
)


@pytest.mark.parametrize("adduct", supported_adducts())
def test_adduct_round_trip(adduct: str) -> None:
    mass = 348.123456
    mz = theoretical_precursor_mz(mass, adduct)
    assert neutral_mass_from_precursor(mz, adduct) == pytest.approx(mass, abs=1e-10)


def test_water_loss_mass_is_monoisotopic() -> None:
    assert WATER_MASS == pytest.approx(18.01056468403, abs=1e-10)


def test_unknown_adduct_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported adduct"):
        neutral_mass_from_precursor(100, "[M+magic]+")


@pytest.mark.parametrize(
    "adduct",
    ["[2M+Na]+", "[2M+H]+", "[2M-H]-", "[M+C2H4O2-H]-", "[M+Br]-", "[M+3H]3+"],
)
def test_formal_training_adduct_round_trip(adduct: str) -> None:
    mass = 512.2345
    precursor = theoretical_precursor_mz(mass, adduct)
    assert neutral_mass_from_precursor(precursor, adduct) == pytest.approx(mass, abs=1e-10)
