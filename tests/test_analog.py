import polars as pl

from casmi.chemistry import exact_mass, theoretical_precursor_mz
from casmi.retrieval import CandidateDatabase, SpectralHit, propagate_analog_candidates
from casmi.spectra import Spectrum


def test_analog_propagation_prefers_structurally_related_candidate() -> None:
    query_mass = exact_mass("CCCO")
    database = CandidateDatabase(
        pl.DataFrame(
            {
                "candidate_id": ["related", "other"],
                "smiles": ["CCCO", "COC(C)C"],
                "canonical_smiles": ["CCCO", "COC(C)C"],
                "inchikey14": ["BDERNNFJNOPAEC", "ZAFNJMIOTHYJRJ"],
                "formula": ["C3H8O", "C3H8O"],
                "exact_mass": [query_mass, query_mass],
                "source": ["test", "test"],
            }
        )
    )
    spectrum = Spectrum(
        molecule_id="q",
        spectrum_id="q1",
        precursor_mz=theoretical_precursor_mz(query_mass, "[M+H]+"),
        adduct="[M+H]+",
        collision_energy=20,
        mz=[31, 43, 59],
        intensity=[0.5, 1, 0.2],
    )
    analog = SpectralHit(
        query_spectrum_id="q1",
        library_spectrum_id="a1",
        candidate_smiles="CCO",
        candidate_inchikey14="LFQSCWFLJHTTHZ",
        direct_cosine=0.9,
        modified_cosine=0.9,
        neutral_loss_cosine=0.9,
        matched_peaks=3,
        explained_query_intensity=1.0,
        mass_error_ppm=None,
        mass_delta=14.0,
        matching_adduct=True,
        collision_energy_difference=0,
    )
    result = propagate_analog_candidates([spectrum], database, [analog])
    assert result[0].smiles == "CCCO"
    assert result[0].analog_score > result[1].analog_score
