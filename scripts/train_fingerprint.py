#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from casmi.chemistry import morgan_fingerprint
from casmi.models import (
    FingerprintFeatureConfig,
    FingerprintModelConfig,
    SpectrumFingerprintMLP,
    require_torch,
    save_fingerprint_model,
    spectrum_fingerprint_features,
)
from casmi.pipeline import CASMIPipeline
from casmi.utils.config import load_config
from casmi.utils.random import seed_everything


def _sample_indices(
    mask: np.ndarray,
    structure_codes: np.ndarray,
    *,
    max_spectra: int,
    max_per_structure: int,
    seed: int,
) -> np.ndarray:
    candidates = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    counts: dict[int, int] = {}
    selected: list[int] = []
    for raw_index in candidates:
        code = int(structure_codes[raw_index])
        count = counts.get(code, 0)
        if count >= max_per_structure:
            continue
        counts[code] = count + 1
        selected.append(int(raw_index))
        if len(selected) >= max_spectra:
            break
    return np.asarray(selected, dtype=np.int64)


def _build_arrays(index, indices: np.ndarray, feature_config, n_bits: int, radius: int):  # type: ignore[no-untyped-def]
    features = np.empty((len(indices), feature_config.input_dim), dtype=np.float16)
    targets = np.empty((len(indices), n_bits), dtype=np.uint8)
    structure_codes = index.structure_codes[indices].astype(np.int64, copy=False)
    target_cache: dict[int, np.ndarray] = {}
    for output_index, raw_index in enumerate(tqdm(indices, desc="featurize", leave=False)):
        spectrum_index = int(raw_index)
        start = int(index.peak_offsets[spectrum_index])
        end = int(index.peak_offsets[spectrum_index + 1])
        adduct = index.adduct_values[int(index.adduct_codes[spectrum_index])]
        polarity_code = int(index.polarities[spectrum_index])
        ionization_mode = "positive" if polarity_code == 1 else "negative" if polarity_code == -1 else None
        collision_energy = float(index.collision_energies[spectrum_index])
        features[output_index] = spectrum_fingerprint_features(
            index.peak_mz[start:end],
            index.peak_intensity[start:end],
            precursor_mz=float(index.precursor_mz[spectrum_index]),
            adduct=adduct,
            collision_energy=collision_energy,
            ionization_mode=ionization_mode,
            config=feature_config,
        ).astype(np.float16)
        structure_code = int(structure_codes[output_index])
        target = target_cache.get(structure_code)
        if target is None:
            target = morgan_fingerprint(
                index.structure_smiles[structure_code], radius=radius, n_bits=n_bits
            )
            target_cache[structure_code] = target
        targets[output_index] = target
    return features, targets, structure_codes


def _predict(model, loader, device, torch_module):  # type: ignore[no-untyped-def]
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch_module.no_grad():
        for features, target in loader:
            logits = model(features.to(device=device, dtype=torch_module.float32))
            probabilities.append(torch_module.sigmoid(logits).cpu().numpy())
            targets.append(target.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def _metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    structure_codes: np.ndarray,
) -> dict[str, float]:
    eps = 1e-7
    clipped = np.clip(probabilities, eps, 1 - eps)
    bce = -np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped))
    binary = probabilities >= 0.5
    intersections = np.count_nonzero(binary & targets.astype(bool), axis=1)
    unions = np.count_nonzero(binary | targets.astype(bool), axis=1)
    tanimoto = np.divide(intersections, unions, out=np.ones_like(intersections, dtype=float), where=unions > 0)

    molecule_scores: list[float] = []
    for code in np.unique(structure_codes):
        rows = structure_codes == code
        pooled = probabilities[rows].mean(axis=0)
        truth = targets[np.flatnonzero(rows)[0]].astype(bool)
        predicted = pooled >= 0.5
        union = np.count_nonzero(predicted | truth)
        molecule_scores.append(np.count_nonzero(predicted & truth) / union if union else 1.0)
    variable_bits = np.flatnonzero((targets.min(axis=0) == 0) & (targets.max(axis=0) == 1))
    if len(variable_bits) > 128:
        variable_bits = variable_bits[
            np.linspace(0, len(variable_bits) - 1, 128, dtype=np.int64)
        ]
    bit_aurocs = [
        roc_auc_score(targets[:, bit], probabilities[:, bit]) for bit in variable_bits
    ]
    return {
        "fingerprint_bce": float(bce),
        "spectrum_binary_tanimoto": float(tanimoto.mean()),
        "molecule_binary_tanimoto": float(np.mean(molecule_scores)),
        "mean_bit_auroc": float(np.mean(bit_aurocs)) if bit_aurocs else float("nan"),
        "auroc_bits": int(len(bit_aurocs)),
        "spectra": int(len(targets)),
        "molecules": int(len(np.unique(structure_codes))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a leakage-safe spectrum-to-Morgan MLP")
    parser.add_argument("--config", type=Path, default=Path("configs/fingerprint.yaml"))
    parser.add_argument("--spectral-index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-spectra", type=int)
    parser.add_argument("--max-early-stopping-spectra", type=int)
    parser.add_argument("--max-report-spectra", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    for name in (
        "epochs",
        "max_train_spectra",
        "max_early_stopping_spectra",
        "max_report_spectra",
    ):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    torch = require_torch()
    started = time.perf_counter()

    feature_config = FingerprintFeatureConfig(
        max_mz=float(config.get("max_mz", 1200.0)),
        bin_width=float(config.get("bin_width", 1.0)),
        bin_aggregation=str(config.get("bin_aggregation", "max")),  # type: ignore[arg-type]
        include_metadata=bool(config.get("include_metadata", True)),
    )
    model_config = FingerprintModelConfig(
        n_bits=int(config.get("n_bits", 2048)),
        hidden_dims=tuple(int(value) for value in config.get("hidden_dims", [512, 256])),
        dropout=float(config.get("dropout", 0.15)),
    )
    pipeline = CASMIPipeline.load(args.spectral_index)
    if pipeline.index is None:
        raise ValueError("Serialized pipeline has no spectral index")
    index = pipeline.index
    splits = pd.read_parquet(args.splits)
    fold_lookup = dict(
        zip(splits.inchikey14.astype(str), splits.fold.astype(int), strict=True)
    )
    structure_folds = np.asarray(
        [fold_lookup.get(index.structure_keys[i], -1) for i in range(len(index.structure_keys))],
        dtype=np.int8,
    )
    if np.any(structure_folds < 0):
        raise ValueError(f"{np.count_nonzero(structure_folds < 0)} index structures are absent from splits")
    spectrum_folds = structure_folds[index.structure_codes]
    report_fold = int(config.get("report_fold", 0))
    early_fold = int(config.get("early_stopping_fold", 1))
    if report_fold == early_fold:
        raise ValueError("report_fold and early_stopping_fold must differ")
    max_per_structure = int(config.get("max_spectra_per_structure", 2))
    train_indices = _sample_indices(
        (spectrum_folds != report_fold) & (spectrum_folds != early_fold),
        index.structure_codes,
        max_spectra=int(config.get("max_train_spectra", 25_000)),
        max_per_structure=max_per_structure,
        seed=seed,
    )
    early_indices = _sample_indices(
        spectrum_folds == early_fold,
        index.structure_codes,
        max_spectra=int(config.get("max_early_stopping_spectra", 5_000)),
        max_per_structure=max_per_structure,
        seed=seed + 1,
    )
    report_indices = _sample_indices(
        spectrum_folds == report_fold,
        index.structure_codes,
        max_spectra=int(config.get("max_report_spectra", 5_000)),
        max_per_structure=max_per_structure,
        seed=seed + 2,
    )
    if not len(train_indices) or not len(early_indices) or not len(report_indices):
        raise ValueError("Train, early-stopping, and report samples must all be non-empty")

    arrays = {}
    for name, selected in (
        ("train", train_indices),
        ("early_stopping", early_indices),
        ("report", report_indices),
    ):
        arrays[name] = _build_arrays(
            index, selected, feature_config, model_config.n_bits, int(config.get("radius", 2))
        )

    class ArrayDataset(torch.utils.data.Dataset):
        def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
            self.features = features
            self.targets = targets

        def __len__(self) -> int:
            return len(self.features)

        def __getitem__(self, item: int):  # type: ignore[no-untyped-def]
            return torch.from_numpy(self.features[item]), torch.from_numpy(self.targets[item])

    batch_size = int(config.get("batch_size", 128))
    loaders = {
        name: torch.utils.data.DataLoader(
            ArrayDataset(values[0], values[1]),
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=0,
        )
        for name, values in arrays.items()
    }
    requested_device = str(config.get("device", "auto"))
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = requested_device
    model = SpectrumFingerprintMLP(feature_config.input_dim, model_config).to(device)
    positive_rate = arrays["train"][1].mean(axis=0)
    positive_weight = np.divide(
        1.0 - positive_rate,
        positive_rate,
        out=np.full_like(positive_rate, float(config.get("max_positive_weight", 50.0))),
        where=positive_rate > 0,
    )
    positive_weight = np.clip(positive_weight, 1.0, float(config.get("max_positive_weight", 50.0)))
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(positive_weight.astype(np.float32)).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = None
    patience = int(config.get("patience", 3))
    stale_epochs = 0
    for epoch in range(1, int(config.get("epochs", 10)) + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for features, target in loaders["train"]:
            features = features.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(features)
            total_rows += len(features)
        early_probabilities, early_targets = _predict(model, loaders["early_stopping"], device, torch)
        early_loss = float(
            -np.mean(
                early_targets * np.log(np.clip(early_probabilities, 1e-7, 1 - 1e-7))
                + (1 - early_targets) * np.log(np.clip(1 - early_probabilities, 1e-7, 1 - 1e-7))
            )
        )
        row = {"epoch": epoch, "train_weighted_bce": total_loss / total_rows, "early_bce": early_loss}
        history.append(row)
        print(json.dumps(row))
        if early_loss < best_loss:
            best_loss = early_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    early_probabilities, early_targets = _predict(model, loaders["early_stopping"], device, torch)
    report_probabilities, report_targets = _predict(model, loaders["report"], device, torch)
    metrics = {
        "early_stopping": _metrics(early_probabilities, early_targets, arrays["early_stopping"][2]),
        "report": _metrics(report_probabilities, report_targets, arrays["report"][2]),
        "best_epoch": min(history, key=lambda row: row["early_bce"])["epoch"],
        "device": device,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save_fingerprint_model(
        model.cpu(),
        args.output / "model.pt",
        feature_config=feature_config,
        model_config=model_config,
        extra={
            "radius": int(config.get("radius", 2)),
            "report_fold": report_fold,
            "early_stopping_fold": early_fold,
        },
    )
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output / "resolved_config.json").write_text(
        json.dumps(
            {
                **config,
                "feature_config": asdict(feature_config),
                "model_config": asdict(model_config),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
