from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

ALIASES: dict[str, tuple[str, ...]] = {
    "molecule_id": ("molecule_id", "compound_id", "molecule"),
    "spectrum_id": ("spectrum_id", "spec_id", "scan_id"),
    "mz": ("ms2_mzs", "mz", "mzs", "peaks_mz"),
    "intensity": (
        "ms2_normalized_intensities",
        "intensity",
        "intensities",
        "peaks_intensity",
    ),
    "precursor_mz": ("precursor_mz", "precursor_mass", "parent_mz"),
    "adduct": ("adduct", "precursor_type"),
    "collision_energy": ("collision_energy_ev", "collision_energy", "ce"),
    "ionization_mode": ("ionization_mode", "polarity", "ion_mode"),
    "smiles": ("normalized_smiles", "smiles", "canonical_smiles"),
    "inchikey14": ("inchikey14", "inchikey_14"),
    "formula": ("molecular_formula", "formula"),
    "exact_mass": ("exact_mass", "monoisotopic_mass"),
    "num_peaks": ("num_peaks", "peak_count"),
}


@dataclass(frozen=True, slots=True)
class ColumnMap:
    molecule_id: str | None
    spectrum_id: str | None
    mz: str
    intensity: str
    precursor_mz: str
    adduct: str | None = None
    collision_energy: str | None = None
    ionization_mode: str | None = None
    smiles: str | None = None
    inchikey14: str | None = None
    formula: str | None = None
    exact_mass: str | None = None
    num_peaks: str | None = None

    def present_columns(self) -> list[str]:
        return [value for value in asdict(self).values() if value is not None]


def _first_present(names: set[str], aliases: Iterable[str]) -> str | None:
    return next((alias for alias in aliases if alias in names), None)


def detect_columns(columns: Iterable[str]) -> ColumnMap:
    names = set(columns)
    found = {name: _first_present(names, aliases) for name, aliases in ALIASES.items()}
    required = ("mz", "intensity", "precursor_mz")
    missing = [name for name in required if found[name] is None]
    if missing:
        raise ValueError(f"Missing required logical columns {missing}; available columns: {sorted(names)}")
    if found["molecule_id"] is None and found["inchikey14"] is None and found["smiles"] is None:
        raise ValueError(
            "No molecule identifier or structure label is available; expected molecule_id, "
            "inchikey14, or a SMILES column"
        )
    return ColumnMap(**found)  # type: ignore[arg-type]


def detect_peak_representation(dtype: pl.DataType, example: object) -> str:
    if isinstance(dtype, pl.List):
        return f"list[{dtype.inner}]"
    if isinstance(dtype, pl.Array):
        return f"array[{dtype.inner}, {dtype.size}]"
    if dtype == pl.String:
        return "string-encoded array"
    if isinstance(example, (list, tuple)):
        return "nested/list array"
    return f"unknown ({dtype})"


def _describe(lf: pl.LazyFrame, column: str) -> dict[str, float | int | None]:
    row = (
        lf.select(
            pl.col(column).min().alias("min"),
            pl.col(column).median().alias("median"),
            pl.col(column).mean().alias("mean"),
            pl.col(column).max().alias("max"),
        )
        .collect(engine="streaming")
        .to_dicts()[0]
    )
    return row


def inspect_parquet(path: str | Path, example_rows: int = 3) -> dict[str, Any]:
    """Inspect parquet lazily and return a JSON-serializable report."""
    source = Path(path)
    lf = pl.scan_parquet(source)
    schema = lf.collect_schema()
    cmap = detect_columns(schema.names())
    if cmap.molecule_id:
        molecule_expression = pl.col(cmap.molecule_id)
        molecule_id_source = cmap.molecule_id
    elif cmap.inchikey14:
        molecule_expression = pl.col(cmap.inchikey14)
        molecule_id_source = f"derived from {cmap.inchikey14}"
    else:
        molecule_expression = pl.col(cmap.smiles)  # type: ignore[arg-type]
        molecule_id_source = f"derived from {cmap.smiles}"
    normalized = lf.with_columns(molecule_expression.alias("__molecule_id"))
    row_count = int(lf.select(pl.len()).collect(engine="streaming").item())
    nulls = lf.select(pl.all().null_count()).collect(engine="streaming").to_dicts()[0]
    examples = lf.select(cmap.present_columns()).head(example_rows).collect().to_dicts()
    mz_example = examples[0][cmap.mz] if examples else None
    intensity_example = examples[0][cmap.intensity] if examples else None

    summarized_examples: list[dict[str, Any]] = []
    for example in examples:
        summarized = dict(example)
        for column in (cmap.mz, cmap.intensity):
            values = summarized.get(column)
            if isinstance(values, (list, tuple)):
                summarized[column] = {"length": len(values), "first_values": list(values[:10])}
        summarized_examples.append(summarized)

    per_molecule = normalized.group_by("__molecule_id").agg(pl.len().alias("spectra"))
    report: dict[str, Any] = {
        "path": str(source),
        "row_count": row_count,
        "columns": schema.names(),
        "dtypes": {name: str(dtype) for name, dtype in schema.items()},
        "null_counts": nulls,
        "detected_columns": asdict(cmap),
        "molecule_id_source": molecule_id_source,
        "spectrum_id_source": cmap.spectrum_id or "derived from parquet row number",
        "peak_representation": {
            "mz": detect_peak_representation(schema[cmap.mz], mz_example),
            "intensity": detect_peak_representation(schema[cmap.intensity], intensity_example),
        },
        "unique_molecule_ids": int(
            normalized.select(pl.col("__molecule_id").n_unique()).collect().item()
        ),
        "spectra_per_molecule": _describe(per_molecule, "spectra"),
        "examples": summarized_examples,
    }
    if cmap.smiles:
        report["unique_structures"] = int(lf.select(pl.col(cmap.smiles).n_unique()).collect().item())
        counts = lf.group_by(cmap.smiles).agg(pl.len().alias("count"))
        report["duplicated_structure_rows"] = int(
            counts.select((pl.col("count") - 1).clip(lower_bound=0).sum()).collect().item()
        )
    if cmap.inchikey14:
        report["unique_inchikey14"] = int(
            lf.select(pl.col(cmap.inchikey14).n_unique()).collect().item()
        )
    if cmap.adduct:
        report["adduct_frequencies"] = (
            lf.group_by(cmap.adduct)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .collect(engine="streaming")
            .to_dicts()
        )
    if cmap.collision_energy and not isinstance(schema[cmap.collision_energy], (pl.List, pl.Array)):
        report["collision_energy_distribution"] = _describe(lf, cmap.collision_energy)
    elif cmap.collision_energy and isinstance(schema[cmap.collision_energy], pl.List):
        report["collision_energy_distribution"] = (
            lf.select(pl.col(cmap.collision_energy).list.explode().alias("ce"))
            .select(
                pl.col("ce").min().alias("min"),
                pl.col("ce").median().alias("median"),
                pl.col("ce").mean().alias("mean"),
                pl.col("ce").max().alias("max"),
            )
            .collect(engine="streaming")
            .to_dicts()[0]
        )
    elif cmap.collision_energy and isinstance(schema[cmap.collision_energy], pl.Array):
        report["collision_energy_distribution"] = (
            lf.select(pl.col(cmap.collision_energy).arr.explode().alias("ce"))
            .select(
                pl.col("ce").min().alias("min"),
                pl.col("ce").median().alias("median"),
                pl.col("ce").mean().alias("mean"),
                pl.col("ce").max().alias("max"),
            )
            .collect(engine="streaming")
            .to_dicts()[0]
        )
    report["precursor_mz_distribution"] = _describe(lf, cmap.precursor_mz)
    if cmap.num_peaks:
        report["peak_count_distribution"] = _describe(lf, cmap.num_peaks)
    elif isinstance(schema[cmap.mz], pl.List):
        peak_lf = lf.select(pl.col(cmap.mz).list.len().alias("num_peaks"))
        report["peak_count_distribution"] = _describe(peak_lf, "num_peaks")
    elif isinstance(schema[cmap.mz], pl.Array):
        peak_lf = lf.select(pl.lit(schema[cmap.mz].size).alias("num_peaks"))
        report["peak_count_distribution"] = _describe(peak_lf, "num_peaks")

    duplicate_keys = [cmap.mz, cmap.intensity, cmap.precursor_mz]
    if cmap.adduct:
        duplicate_keys.append(cmap.adduct)
    unique_hashes = lf.select(pl.struct(duplicate_keys).hash().n_unique()).collect(engine="streaming").item()
    report["duplicated_spectrum_count"] = row_count - int(unique_hashes)
    return report


def report_to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=str)
