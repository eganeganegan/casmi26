#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski
from tqdm import tqdm

from casmi.retrieval import CandidateDatabase


def _disable_rdkit_logs() -> None:
    rdBase.DisableLog("rdApp.*")


def _descriptor_values(row: tuple[object, object]) -> tuple[object, ...]:
    key, smiles = row
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError("invalid SMILES")
        heavy_atoms = max(1, mol.GetNumHeavyAtoms())
        return (
            str(key),
            float(Lipinski.NumHeteroatoms(mol)),
            float(Lipinski.RingCount(mol)),
            float(sum(atom.GetIsAromatic() for atom in mol.GetAtoms()) / heavy_atoms),
            float(Descriptors.MolLogP(mol)),
            False,
        )
    except (ValueError, RuntimeError):
        return str(key), np.nan, np.nan, np.nan, np.nan, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reusable RDKit candidate descriptor features")
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--worker-chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.workers <= 0 or args.worker_chunk_size <= 0:
        raise ValueError("workers and worker-chunk-size must be positive")
    started = time.perf_counter()
    database = CandidateDatabase.from_parquet(args.candidate_db)
    schema = pa.schema(
        [
            ("inchikey14", pa.string()),
            ("candidate_heteroatom_count", pa.float64()),
            ("candidate_ring_count", pa.float64()),
            ("candidate_aromatic_fraction", pa.float64()),
            ("candidate_logp", pa.float64()),
        ]
    )
    rows: list[dict[str, object]] = []
    invalid = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        dir=args.output.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()

    def flush(writer: pq.ParquetWriter) -> None:
        nonlocal rows
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            rows = []

    try:
        with pq.ParquetWriter(
            temporary_path, schema, compression="zstd", write_statistics=True
        ) as writer, mp.Pool(
            processes=args.workers, initializer=_disable_rdkit_logs
        ) as pool:
            source_rows = database.frame.select("inchikey14", "smiles").iter_rows()
            values_iterator = pool.imap(
                _descriptor_values,
                source_rows,
                chunksize=args.worker_chunk_size,
            )
            for key, heteroatoms, rings, aromatic_fraction, logp, is_invalid in tqdm(
                values_iterator,
                total=len(database),
                desc="descriptors",
                unit="structure",
            ):
                invalid += int(is_invalid)
                rows.append(
                    {
                        "inchikey14": str(key),
                        "candidate_heteroatom_count": heteroatoms,
                        "candidate_ring_count": rings,
                        "candidate_aromatic_fraction": aromatic_fraction,
                        "candidate_logp": logp,
                    }
                )
                if len(rows) >= args.batch_size:
                    flush(writer)
            flush(writer)
        temporary_path.replace(args.output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    report = {
        "structures": len(database),
        "invalid": invalid,
        "runtime_seconds": time.perf_counter() - started,
        "output": str(args.output),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
