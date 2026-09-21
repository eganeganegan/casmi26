import pytest

from casmi.chemistry import exact_mass, theoretical_precursor_mz
from casmi.pipeline import CASMIPipeline
from casmi.retrieval import CompactSpectralLibraryIndex, SpectralLibraryIndex
from casmi.spectra import Spectrum


def _spectrum(molecule: str, spectrum: str, smiles: str | None, peaks: list[float]) -> Spectrum:
    structure = smiles or "CCO"
    mass = exact_mass(structure)
    return Spectrum(
        molecule_id=molecule,
        spectrum_id=spectrum,
        precursor_mz=theoretical_precursor_mz(mass, "[M+H]+"),
        adduct="[M+H]+",
        collision_energy=20,
        mz=peaks,
        intensity=[1.0, 0.5, 0.25],
        ionization_mode="positive",
        smiles=smiles,
        exact_mass=mass if smiles else None,
    )


def test_pipeline_aggregates_molecule_spectra() -> None:
    library = [
        _spectrum("lib_a", "a1", "CCO", [31, 45, 47]),
        _spectrum("lib_a", "a2", "CCO", [31, 45, 47]),
        _spectrum("lib_b", "b1", "COC", [15, 29, 45]),
    ]
    queries = [
        _spectrum("query", "q1", None, [31, 45, 47]),
        _spectrum("query", "q2", None, [31.001, 45, 47]),
    ]
    result = CASMIPipeline(top_spectra=10).fit(library).predict_molecule(queries)
    assert result.ranked_candidates[0] == "CCO"
    assert result.evidence[0].supporting_spectra == 2
    assert result.evidence[0].supporting_library_spectra == 2


def test_compact_index_matches_object_index_and_round_trips(tmp_path) -> None:
    library = [
        _spectrum("lib_a", "a1", "CCO", [31, 45, 47]),
        _spectrum("lib_b", "b1", "COC", [15, 29, 45]),
    ]
    query = _spectrum("query", "q1", None, [31, 45, 47])
    object_hits = SpectralLibraryIndex(library).search(query, top_n=2)
    compact = CompactSpectralLibraryIndex(
        [
            _spectrum("lib_a", "a1", "CCO", [31, 45, 47]),
            _spectrum("lib_b", "b1", "COC", [15, 29, 45]),
        ]
    )
    compact_hits = compact.search(query, top_n=2)
    assert compact.peak_count == 6
    assert compact.storage_nbytes > 0
    assert not hasattr(compact, "spectra")
    assert [hit.candidate_inchikey14 for hit in compact_hits] == [
        hit.candidate_inchikey14 for hit in object_hits
    ]
    assert compact_hits[0].score == pytest.approx(object_hits[0].score, abs=1e-6)

    path = tmp_path / "compact.joblib"
    compact.save(path)
    loaded = CompactSpectralLibraryIndex.load(path)
    assert loaded.search(query, top_n=1)[0].candidate_smiles == "CCO"


def test_library_search_can_exclude_the_query_spectrum() -> None:
    library = [
        _spectrum("lib_a", "a1", "CCO", [31, 45, 47]),
        _spectrum("lib_a", "a2", "CCO", [31, 44, 47]),
        _spectrum("lib_b", "b1", "COC", [15, 29, 45]),
    ]
    query = _spectrum("query", "a1", None, [31, 45, 47])
    for index in (SpectralLibraryIndex(library), CompactSpectralLibraryIndex(library)):
        hits = index.search(
            query,
            top_n=10,
            exclude_library_spectrum_ids={"a1"},
        )
        assert hits
        assert all(hit.library_spectrum_id != "a1" for hit in hits)
        assert hits[0].candidate_smiles == "CCO"
