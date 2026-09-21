import pandas as pd

from scripts.blend_spectral_predictions import blend_spectral_gate


def test_spectral_gate_prepends_only_confident_unique_hit() -> None:
    ranked = pd.DataFrame(
        {
            "molecule_id": ["a", "a", "b", "b"],
            "rank": [1, 2, 1, 2],
            "smiles": ["CC", "CCC", "O", "CO"],
            "inchikey14": ["A" * 14, "B" * 14, "C" * 14, "D" * 14],
            "ranker_score": [0.8, 0.7, 0.9, 0.6],
        }
    )
    spectral = pd.DataFrame(
        {
            "molecule_id": ["a", "b"],
            "rank": [1, 1],
            "smiles": ["CCC", "N"],
            "inchikey14": ["B" * 14, "E" * 14],
            "max_cosine": [0.99, 0.85],
            "explained_query_intensity": [0.95, 0.95],
            "matched_peaks": [12, 12],
        }
    )
    output = blend_spectral_gate(ranked, spectral)
    a = output.loc[output.molecule_id.eq("a")]
    b = output.loc[output.molecule_id.eq("b")]
    assert a.inchikey14.tolist() == ["B" * 14, "A" * 14]
    assert a["rank"].tolist() == [1, 2]
    assert a.blend_source.tolist() == ["spectral_gate", "ranker"]
    assert b.inchikey14.tolist() == ["C" * 14, "D" * 14]
