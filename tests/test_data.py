from pathlib import Path

import pandas as pd

from casmi.data import detect_columns, inspect_parquet, iter_parquet_spectra

TRAIN_COLUMNS = [
    "normalized_smiles",
    "inchikey14",
    "molecular_formula",
    "adduct",
    "precursor_mz",
    "ms2_mzs",
    "ms2_normalized_intensities",
]


def test_training_schema_without_explicit_ids_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "normalized_smiles": ["CCO", "CCO", "CCC"],
            "inchikey14": ["LFQSCWFLJHTTHZ", "LFQSCWFLJHTTHZ", "ATUOYWHBWRKTHZ"],
            "molecular_formula": ["C2H6O", "C2H6O", "C3H8"],
            "adduct": ["[M+H]+"] * 3,
            "precursor_mz": [47.049, 47.049, 45.070],
            "ms2_mzs": [[31.0, 45.0], [31.0], [29.0]],
            "ms2_normalized_intensities": [[1.0, 0.5], [1.0], [1.0]],
        }
    ).to_parquet(path, index=False)

    mapping = detect_columns(TRAIN_COLUMNS)
    assert mapping.molecule_id is None
    assert mapping.spectrum_id is None

    spectra = list(iter_parquet_spectra(path, batch_size=2))
    assert [s.molecule_id for s in spectra] == [
        "LFQSCWFLJHTTHZ",
        "LFQSCWFLJHTTHZ",
        "ATUOYWHBWRKTHZ",
    ]
    assert [s.spectrum_id for s in spectra] == [
        "train_row_000000000",
        "train_row_000000001",
        "train_row_000000002",
    ]
    filtered = list(iter_parquet_spectra(path, batch_size=1, molecule_ids={"ATUOYWHBWRKTHZ"}))
    assert [s.molecule_id for s in filtered] == ["ATUOYWHBWRKTHZ"]
    assert filtered[0].spectrum_id == "train_filtered_row_000000000"

    report = inspect_parquet(path, example_rows=1)
    assert report["unique_molecule_ids"] == 2
    assert report["molecule_id_source"] == "derived from inchikey14"
    assert report["examples"][0]["ms2_mzs"] == {
        "length": 2,
        "first_values": [31.0, 45.0],
    }
