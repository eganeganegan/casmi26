#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.data import iter_parquet_spectra
from casmi.pipeline import CASMIPipeline
from casmi.spectra import Spectrum
from casmi.validation import evaluate_candidate_recall, evaluate_mrr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a strictly structure-disjoint retrieval fold")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-validation-molecules", type=int)
    parser.add_argument("--max-library-spectra", type=int)
    parser.add_argument("--top-candidates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started = time.perf_counter()
    split_frame = pd.read_parquet(args.splits)
    if split_frame.groupby("inchikey14")["fold"].nunique().max() > 1:
        raise AssertionError("Fold file contains structure leakage")
    fold_by_molecule = dict(
        zip(split_frame.molecule_id.astype(str), split_frame.fold, strict=True)
    )
    validation_ids = set(split_frame.loc[split_frame.fold == args.fold, "molecule_id"].astype(str))
    if args.max_validation_molecules and len(validation_ids) > args.max_validation_molecules:
        rng = np.random.default_rng(args.seed)
        validation_ids = set(
            rng.choice(
                np.asarray(sorted(validation_ids)),
                size=args.max_validation_molecules,
                replace=False,
            ).tolist()
        )

    validation_keys = set(split_frame.loc[split_frame.fold == args.fold, "inchikey14"])

    def iter_library() -> Iterator[Spectrum]:
        library_count = 0
        for spectrum in iter_parquet_spectra(args.train):
            fold = fold_by_molecule.get(spectrum.molecule_id)
            if fold is None:
                raise KeyError(f"Molecule {spectrum.molecule_id!r} is missing from the split file")
            if fold == args.fold:
                continue
            if spectrum.inchikey14 in validation_keys:
                raise AssertionError(f"Retrieval library leakage detected for {spectrum.inchikey14}")
            library_count += 1
            yield spectrum
            if args.max_library_spectra is not None and library_count >= args.max_library_spectra:
                return

    pipeline = CASMIPipeline().fit(iter_library())
    assert pipeline.index is not None
    validation = defaultdict(list)
    truth: dict[str, str] = {}
    for spectrum in iter_parquet_spectra(args.train, molecule_ids=validation_ids):
        if fold_by_molecule.get(spectrum.molecule_id) != args.fold:
            raise AssertionError(f"Validation filtering crossed folds for {spectrum.molecule_id}")
        truth[spectrum.molecule_id] = spectrum.molecule_id
        validation[spectrum.molecule_id].append(
            replace(
                spectrum,
                smiles=None,
                canonical_smiles=None,
                inchikey14=None,
                exact_mass=None,
                molecular_formula=None,
            )
        )
    predictions = {
        molecule_id: pipeline.predict_molecule(spectra, max_candidates=args.top_candidates)
        for molecule_id, spectra in validation.items()
    }
    ranked = {
        key: [candidate.inchikey14 for candidate in value.evidence]
        for key, value in predictions.items()
    }
    metrics = evaluate_mrr(truth, ranked).to_dict()
    metrics.update(evaluate_candidate_recall(truth, ranked, (25, 100, 500)))
    elapsed = time.perf_counter() - started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = peak_rss / (1024**2) if sys.platform == "darwin" else peak_rss / 1024
    report = {
        **metrics,
        "fold": args.fold,
        "library_spectra": len(pipeline.index),
        "library_peaks": pipeline.index.peak_count,
        "index_storage_mb": pipeline.index.storage_nbytes / (1024**2),
        "validation_molecules": len(validation),
        "seed": args.seed,
        "runtime_seconds": elapsed,
        "peak_rss_mb": rss,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        rows = []
        for molecule_id, prediction in predictions.items():
            for rank, candidate in enumerate(prediction.evidence, 1):
                rows.append({"molecule_id": molecule_id, "rank": rank, **candidate.to_dict()})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(args.output, index=False)
        metrics_path = args.output.parent / "metrics.json"
        metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
