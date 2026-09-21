#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.data import iter_parquet_spectra
from casmi.models import (
    featurize_spectrum,
    load_fingerprint_model,
    pool_fingerprint_probabilities,
    require_torch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict molecule-level Morgan probabilities")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pooling", choices=["mean", "confidence_weighted"], default="mean")
    args = parser.parse_args()
    torch = require_torch()
    model, feature_config, _, _ = load_fingerprint_model(args.model)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    spectra = list(iter_parquet_spectra(args.input))
    with torch.no_grad():
        for start in range(0, len(spectra), args.batch_size):
            batch = spectra[start : start + args.batch_size]
            features = np.stack(
                [featurize_spectrum(spectrum, feature_config=feature_config) for spectrum in batch]
            )
            logits = model(torch.from_numpy(features).to(device=device, dtype=torch.float32))
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for spectrum, probability in zip(batch, probabilities, strict=True):
                grouped[spectrum.molecule_id].append(probability)
    rows = [
        {
            "molecule_id": molecule_id,
            "fingerprint_probability": pool_fingerprint_probabilities(
                np.stack(values), method=args.pooling
            ).tolist(),
            "num_spectra": len(values),
        }
        for molecule_id, values in sorted(grouped.items())
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(f"wrote fingerprint probabilities for {len(rows):,} molecules to {args.output}")


if __name__ == "__main__":
    main()
