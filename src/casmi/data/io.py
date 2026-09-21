from __future__ import annotations

from collections.abc import Collection, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from casmi.data.schema import ColumnMap, detect_columns
from casmi.spectra.preprocessing import parse_peak_array
from casmi.spectra.types import Spectrum


def iter_parquet_spectra(
    path: str | Path,
    batch_size: int = 4096,
    molecule_ids: Collection[str] | None = None,
) -> Iterator[Spectrum]:
    """Stream normalized Spectrum records without loading the parquet file in memory."""
    parquet = pq.ParquetFile(path)
    cmap: ColumnMap = detect_columns(parquet.schema_arrow.names)
    columns = cmap.present_columns()
    if molecule_ids is not None:
        molecule_column = cmap.molecule_id or cmap.inchikey14 or cmap.smiles
        if molecule_column is None:  # Protected by detect_columns.
            raise ValueError("Cannot filter spectra without a molecule identity column")
        value_set = pa.array(sorted(str(value) for value in molecule_ids))
        id_prefix = f"{Path(path).stem}_filtered_row"
    else:
        molecule_column = None
        value_set = None
        id_prefix = f"{Path(path).stem}_row"
    row_number = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        if molecule_column is not None and value_set is not None:
            identity = batch.column(batch.schema.get_field_index(molecule_column))
            batch = batch.filter(pc.is_in(identity, value_set=value_set))
            if batch.num_rows == 0:
                continue
        for row in batch.to_pylist():
            if cmap.molecule_id:
                molecule_id = str(row[cmap.molecule_id])
            elif cmap.inchikey14:
                molecule_id = str(row[cmap.inchikey14])
            elif cmap.smiles:
                molecule_id = str(row[cmap.smiles])
            else:  # Protected by detect_columns; retained for type narrowing.
                raise ValueError("Cannot derive molecule_id")
            spectrum_id = (
                str(row[cmap.spectrum_id])
                if cmap.spectrum_id
                else f"{id_prefix}_{row_number:09d}"
            )
            yield Spectrum(
                molecule_id=molecule_id,
                spectrum_id=spectrum_id,
                precursor_mz=float(row[cmap.precursor_mz]),
                adduct=str(row[cmap.adduct]) if cmap.adduct and row[cmap.adduct] is not None else "",
                collision_energy=row[cmap.collision_energy] if cmap.collision_energy else None,
                mz=parse_peak_array(row[cmap.mz]),
                intensity=parse_peak_array(row[cmap.intensity]),
                ionization_mode=(
                    str(row[cmap.ionization_mode])
                    if cmap.ionization_mode and row[cmap.ionization_mode] is not None
                    else None
                ),
                smiles=(str(row[cmap.smiles]) if cmap.smiles and row[cmap.smiles] is not None else None),
                canonical_smiles=(
                    str(row[cmap.smiles]) if cmap.smiles and row[cmap.smiles] is not None else None
                ),
                inchikey14=(
                    str(row[cmap.inchikey14])
                    if cmap.inchikey14 and row[cmap.inchikey14] is not None
                    else None
                ),
                molecular_formula=(
                    str(row[cmap.formula]) if cmap.formula and row[cmap.formula] is not None else None
                ),
                exact_mass=(
                    float(row[cmap.exact_mass])
                    if cmap.exact_mass and row[cmap.exact_mass] is not None
                    else None
                ),
            )
            row_number += 1
