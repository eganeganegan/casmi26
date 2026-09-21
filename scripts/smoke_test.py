#!/usr/bin/env python3
from __future__ import annotations

import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from casmi.chemistry import exact_mass, smiles_to_inchikey14, theoretical_precursor_mz
from casmi.data import inspect_parquet, iter_parquet_spectra
from casmi.inference import make_submission
from casmi.pipeline import CASMIPipeline
from casmi.ranking import score_candidate_fragments
from casmi.spectra import Spectrum
from casmi.validation import evaluate_candidate_recall, evaluate_mrr


def synthetic_spectra() -> list[Spectrum]:
    structures = {
        "ethanol": ("CCO", [31.018, 45.034, 47.049]),
        "phenol": ("Oc1ccccc1", [39.023, 65.039, 77.039, 94.042]),
        "caffeine": ("Cn1c(=O)c2c(ncn2C)n(C)c1=O", [55.030, 82.041, 109.052, 138.066]),
    }
    rows: list[Spectrum] = []
    for i, (name, (smiles, peaks)) in enumerate(structures.items()):
        mass = exact_mass(smiles)
        for replicate in range(2):
            rows.append(
                Spectrum(
                    molecule_id=f"train_{name}",
                    spectrum_id=f"tr_{i}_{replicate}",
                    precursor_mz=theoretical_precursor_mz(mass, "[M+H]+"),
                    adduct="[M+H]+",
                    collision_energy=[20 + 10 * replicate],
                    mz=peaks,
                    intensity=[0.3, 0.7, 1.0, 0.5][: len(peaks)],
                    ionization_mode="positive",
                    smiles=smiles,
                    inchikey14=smiles_to_inchikey14(smiles),
                    exact_mass=mass,
                )
            )
    return rows


def main() -> None:
    start = time.perf_counter()
    library = synthetic_spectra()
    pipeline = CASMIPipeline(top_spectra=20).fit(library)
    query_smiles = "Oc1ccccc1"
    query = Spectrum(
        molecule_id="query_phenol",
        spectrum_id="query_0",
        precursor_mz=theoretical_precursor_mz(exact_mass(query_smiles), "[M+H]+"),
        adduct="[M+H]+",
        collision_energy=[25],
        mz=[39.0231, 65.039, 77.0391, 94.042],
        intensity=[0.29, 0.72, 1.0, 0.48],
        ionization_mode="positive",
    )
    prediction = pipeline.predict_molecule([query])
    predictions = {query.molecule_id: prediction.ranked_candidates}
    truth = {query.molecule_id: query_smiles}
    metrics = evaluate_mrr(truth, predictions).to_dict()
    recall = evaluate_candidate_recall(truth, predictions, (25, 100, 500))
    fragment_features = score_candidate_fragments(query_smiles, [query])
    assert metrics["mrr_at_25"] == 1.0
    assert fragment_features["fragment_scored"] == 1.0

    with tempfile.TemporaryDirectory(prefix="casmi-smoke-") as temp:
        root = Path(temp)
        parquet_path = root / "test.parquet"
        pd.DataFrame(
            {
                "molecule_id": [query.molecule_id],
                "spectrum_id": [query.spectrum_id],
                "ms2_mzs": [query.mz.tolist()],
                "ms2_normalized_intensities": [query.intensity.tolist()],
                "precursor_mz": [query.precursor_mz],
                "adduct": [query.adduct],
                "collision_energy_ev": [query.collision_energy],
                "ionization_mode": [query.ionization_mode],
            }
        ).to_parquet(parquet_path, index=False)
        schema_report = inspect_parquet(parquet_path, example_rows=1)
        streamed = list(iter_parquet_spectra(parquet_path))
        assert len(streamed) == 1
        submission_path = root / "submission.csv"
        make_submission(predictions, submission_path, expected_molecule_ids=[query.molecule_id])
        assert submission_path.exists()

    elapsed = time.perf_counter() - start
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak_rss / (1024**2) if sys.platform == "darwin" else peak_rss / 1024
    print(
        json.dumps(
            {
                "status": "ok",
                "schema_columns": schema_report["columns"],
                "metrics": metrics,
                "candidate_recall": recall,
                "fragment_intensity_fraction": fragment_features[
                    "fragment_intensity_fraction_max"
                ],
                "runtime_seconds": round(elapsed, 4),
                "peak_rss_mb": round(peak_mb, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
