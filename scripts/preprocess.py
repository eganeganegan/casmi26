#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from casmi.chemistry import canonicalize_smiles, exact_mass, smiles_to_inchikey14
from casmi.data.schema import detect_columns
from casmi.spectra import SpectrumPreprocessingConfig, clean_spectrum


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean peak arrays in bounded-memory parquet batches")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--min-relative-intensity", type=float, default=0.01)
    parser.add_argument("--top-n", type=int, default=256)
    parser.add_argument("--intensity-transform", choices=["raw", "sqrt", "log1p"], default="sqrt")
    args = parser.parse_args()
    source = pq.ParquetFile(args.input)
    cmap = detect_columns(source.schema_arrow.names)
    cfg = SpectrumPreprocessingConfig(
        min_relative_intensity=args.min_relative_intensity,
        top_n=args.top_n,
        intensity_transform=args.intensity_transform,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        total_batches = math.ceil(source.metadata.num_rows / args.batch_size)
        for batch in tqdm(source.iter_batches(batch_size=args.batch_size), total=total_batches):
            rows = batch.to_pylist()
            for row in rows:
                mz, intensity = clean_spectrum(
                    row[cmap.mz],
                    row[cmap.intensity],
                    precursor_mz=row[cmap.precursor_mz],
                    config=cfg,
                )
                row[cmap.mz] = mz.tolist()
                row[cmap.intensity] = intensity.tolist()
                row["num_peaks"] = len(mz)
                if cmap.smiles and row.get(cmap.smiles):
                    smiles = str(row[cmap.smiles])
                    row["canonical_smiles"] = canonicalize_smiles(smiles)
                    row["inchikey14"] = smiles_to_inchikey14(smiles)
                    row["exact_mass"] = exact_mass(smiles)
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    print(f"wrote cleaned parquet to {args.output}")


if __name__ == "__main__":
    main()
