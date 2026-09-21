from __future__ import annotations

import gzip
import pickle
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

from casmi.chemistry import smiles_to_inchikey14

REQUIRED_COLUMNS = (
    "candidate_id",
    "smiles",
    "canonical_smiles",
    "inchikey14",
    "formula",
    "exact_mass",
    "source",
)


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    candidate_id: str
    smiles: str
    canonical_smiles: str
    inchikey14: str
    formula: str | None
    exact_mass: float
    source: str
    mass_error_ppm: float | None = None


@dataclass(frozen=True, slots=True)
class MassChannelCandidate:
    """A mass candidate annotated with every tolerance channel that retrieved it."""

    candidate: CandidateRecord
    min_tolerance_ppm: float
    channels_ppm: tuple[float, ...]


class CandidateDatabase:
    """Compact offline candidate table with sorted mass and formula indexes."""

    def __init__(self, frame: pl.DataFrame) -> None:
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Candidate database is missing columns: {sorted(missing)}")
        self.frame = frame.select(REQUIRED_COLUMNS)
        if self.frame["inchikey14"].n_unique() != len(self.frame):
            raise ValueError("Candidate database must be deduplicated by inchikey14")
        masses = self.frame["exact_mass"].to_numpy().astype(np.float64, copy=False)
        finite = np.flatnonzero(np.isfinite(masses))
        self.masses = masses
        self.mass_order = finite[np.argsort(masses[finite], kind="stable")]
        self.sorted_masses = masses[self.mass_order]
        self._formula_index: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.frame)

    @classmethod
    def from_parquet(cls, path: str | Path) -> CandidateDatabase:
        return cls(pl.read_parquet(path, columns=list(REQUIRED_COLUMNS)))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.frame.write_parquet(destination, compression="zstd", statistics=True)

    def _records(self, indices: np.ndarray, query_mass: float | None = None) -> list[CandidateRecord]:
        if not len(indices):
            return []
        rows = self.frame[indices.tolist()].to_dicts()
        records = []
        for row in rows:
            ppm = None
            if query_mass is not None:
                ppm = 1_000_000.0 * (query_mass - float(row["exact_mass"])) / float(row["exact_mass"])
            records.append(
                CandidateRecord(
                    candidate_id=str(row["candidate_id"]),
                    smiles=str(row["smiles"]),
                    canonical_smiles=str(row["canonical_smiles"]),
                    inchikey14=str(row["inchikey14"]),
                    formula=str(row["formula"]) if row["formula"] is not None else None,
                    exact_mass=float(row["exact_mass"]),
                    source=str(row["source"]),
                    mass_error_ppm=ppm,
                )
            )
        return records

    def query_mass(self, mass: float, ppm: float, limit: int | None = None) -> list[CandidateRecord]:
        if mass <= 0 or ppm < 0:
            raise ValueError("mass must be positive and ppm must be non-negative")
        delta = mass * ppm * 1e-6
        left = np.searchsorted(self.sorted_masses, mass - delta, side="left")
        right = np.searchsorted(self.sorted_masses, mass + delta, side="right")
        indices = self.mass_order[left:right]
        indices = indices[np.argsort(np.abs(self.masses[indices] - mass), kind="stable")]
        if limit is not None:
            indices = indices[:limit]
        return self._records(indices, query_mass=mass)

    def query_mass_channels(
        self,
        mass: float,
        tolerances_ppm: Iterable[float],
        limit: int | None = None,
    ) -> list[MassChannelCandidate]:
        """Query nested mass windows once and retain channel-membership features.

        Results remain ordered by absolute mass error. ``limit`` applies to the
        union returned from the widest channel, not independently per channel.
        """
        tolerances = tuple(sorted(set(float(value) for value in tolerances_ppm)))
        if not tolerances:
            raise ValueError("At least one mass tolerance is required")
        if any(value < 0 or not np.isfinite(value) for value in tolerances):
            raise ValueError("Mass tolerances must be finite and non-negative")
        candidates = self.query_mass(mass, tolerances[-1], limit=limit)
        output: list[MassChannelCandidate] = []
        for candidate in candidates:
            absolute_delta = abs(candidate.exact_mass - mass)
            channels = tuple(
                tolerance
                for tolerance in tolerances
                if absolute_delta <= mass * tolerance * 1e-6
            )
            if channels:
                output.append(
                    MassChannelCandidate(
                        candidate=candidate,
                        min_tolerance_ppm=channels[0],
                        channels_ppm=channels,
                    )
                )
        return output

    def _ensure_formula_index(self) -> dict[str, np.ndarray]:
        if self._formula_index is None:
            grouped: dict[str, list[int]] = {}
            for index, formula in enumerate(self.frame["formula"].to_list()):
                if formula is not None:
                    grouped.setdefault(str(formula), []).append(index)
            self._formula_index = {
                formula: np.asarray(indices, dtype=np.int64) for formula, indices in grouped.items()
            }
        return self._formula_index

    def query_formula(self, formula: str, limit: int | None = None) -> list[CandidateRecord]:
        indices = self._ensure_formula_index().get(formula, np.empty(0, dtype=np.int64))
        if limit is not None:
            indices = indices[:limit]
        return self._records(indices)

    def query_mass_and_formula(
        self, mass: float, ppm: float, formula: str, limit: int | None = None
    ) -> list[CandidateRecord]:
        mass_records = self.query_mass(mass, ppm)
        records = [record for record in mass_records if record.formula == formula]
        return records[:limit] if limit is not None else records


def build_candidate_database(
    input_path: str | Path,
    output_path: str | Path,
    *,
    smiles_column: str,
    exact_mass_column: str,
    source: str,
    candidate_id_column: str | None = None,
    inchikey_column: str | None = None,
    formula_column: str | None = None,
    separator: str | None = None,
) -> dict[str, int]:
    """Normalize a CSV/TSV/parquet structure source into the common compact parquet schema."""
    source_path = Path(input_path)
    if source_path.suffix.lower() == ".parquet":
        lazy = pl.scan_parquet(source_path)
    elif source_path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        actual_separator = separator or ("\t" if source_path.suffix.lower() == ".tsv" else ",")
        lazy = pl.scan_csv(source_path, separator=actual_separator, infer_schema_length=10_000)
    else:
        raise ValueError(f"Unsupported candidate input type: {source_path.suffix}")
    available = set(lazy.collect_schema().names())
    required_source = {smiles_column, exact_mass_column}
    for optional in (candidate_id_column, inchikey_column, formula_column):
        if optional:
            required_source.add(optional)
    missing = required_source - available
    if missing:
        raise ValueError(f"Candidate source is missing columns: {sorted(missing)}")
    if inchikey_column is None:
        raise ValueError(
            "An InChIKey column is required for streaming builds; preprocess sources lacking one "
            "with RDKit before calling this builder"
        )

    raw_count = int(lazy.select(pl.len()).collect(engine="streaming").item())
    candidate_id = (
        pl.col(candidate_id_column).cast(pl.String)
        if candidate_id_column
        else pl.col(inchikey_column).cast(pl.String)
    )
    formula = pl.col(formula_column).cast(pl.String) if formula_column else pl.lit(None, dtype=pl.String)
    normalized = (
        lazy.select(
            candidate_id.alias("candidate_id"),
            pl.col(smiles_column).cast(pl.String).alias("smiles"),
            pl.col(smiles_column).cast(pl.String).alias("canonical_smiles"),
            pl.col(inchikey_column).cast(pl.String).str.slice(0, 14).alias("inchikey14"),
            formula.alias("formula"),
            pl.col(exact_mass_column).cast(pl.Float64, strict=False).alias("exact_mass"),
            pl.lit(source).alias("source"),
        )
        .filter(
            pl.col("smiles").is_not_null()
            & (pl.col("smiles").str.len_chars() > 0)
            & pl.col("inchikey14").is_not_null()
            & (pl.col("inchikey14").str.len_chars() == 14)
            & pl.col("exact_mass").is_finite()
            & (pl.col("exact_mass") > 0)
        )
        .unique(subset=["inchikey14"], keep="first", maintain_order=False)
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized.sink_parquet(destination, compression="zstd", statistics=True, mkdir=True)
    output_count = int(pl.scan_parquet(destination).select(pl.len()).collect().item())
    return {"input_rows": raw_count, "output_rows": output_count, "duplicates_or_invalid": raw_count - output_count}


def build_sdf_candidate_database(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source: str,
    id_property: str | None = None,
    inchikey_property: str | None = None,
    batch_size: int = 50_000,
    max_records: int | None = None,
) -> dict[str, int]:
    """Stream an SDF/SDF.GZ structure dump into the compact candidate schema."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")
    source_path = Path(input_path)
    if not (
        source_path.suffix.lower() == ".sdf"
        or source_path.name.lower().endswith(".sdf.gz")
    ):
        raise ValueError("SDF input must end with .sdf or .sdf.gz")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("candidate_id", pa.string()),
            ("smiles", pa.string()),
            ("canonical_smiles", pa.string()),
            ("inchikey14", pa.string()),
            ("formula", pa.string()),
            ("exact_mass", pa.float64()),
            ("source", pa.string()),
        ]
    )
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    input_records = 0
    invalid = 0
    duplicates = 0
    written = 0
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()

    def flush(writer: pq.ParquetWriter) -> int:
        nonlocal rows
        if not rows:
            return 0
        table = pa.Table.from_pylist(rows, schema=schema)
        writer.write_table(table)
        count = len(rows)
        rows = []
        return count

    opener = gzip.open if source_path.name.lower().endswith(".gz") else open
    try:
        with (
            opener(source_path, "rb") as handle,
            pq.ParquetWriter(
                temporary_path, schema, compression="zstd", write_statistics=True
            ) as writer,
            rdBase.BlockLogs(),
        ):
            supplier = Chem.ForwardSDMolSupplier(handle, sanitize=True, removeHs=True)
            for mol in tqdm(supplier, desc="SDF candidates", unit="record"):
                if max_records is not None and input_records >= max_records:
                    break
                input_records += 1
                if mol is None:
                    invalid += 1
                    continue
                try:
                    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                    raw_key = (
                        mol.GetProp(inchikey_property)
                        if inchikey_property and mol.HasProp(inchikey_property)
                        else Chem.MolToInchiKey(mol)
                    )
                    key = str(raw_key).split("-", maxsplit=1)[0]
                    if not smiles or len(key) != 14:
                        raise ValueError("missing canonical structure identity")
                    mass = float(rdMolDescriptors.CalcExactMolWt(mol))
                    if not np.isfinite(mass) or mass <= 0:
                        raise ValueError("invalid exact mass")
                    formula = str(rdMolDescriptors.CalcMolFormula(mol))
                except (ValueError, RuntimeError):
                    invalid += 1
                    continue
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                candidate_id = (
                    mol.GetProp(id_property)
                    if id_property and mol.HasProp(id_property)
                    else mol.GetProp("_Name")
                    if mol.HasProp("_Name")
                    else key
                )
                rows.append(
                    {
                        "candidate_id": str(candidate_id),
                        "smiles": smiles,
                        "canonical_smiles": smiles,
                        "inchikey14": key,
                        "formula": formula,
                        "exact_mass": mass,
                        "source": source,
                    }
                )
                if len(rows) >= batch_size:
                    written += flush(writer)
            written += flush(writer)
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return {
        "input_records": input_records,
        "output_rows": written,
        "duplicates": duplicates,
        "invalid": invalid,
    }


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Load simple NumPy metadata without permitting arbitrary pickle globals."""

    _ALLOWED = {
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str):  # type: ignore[no-untyped-def]
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"Blocked unsafe pickle global {module}.{name}")
        return super().find_class(module, name)


def build_numpy_candidate_database(
    metadata_path: str | Path,
    mass_path: str | Path,
    output_path: str | Path,
    *,
    source: str,
) -> dict[str, int]:
    """Build from a keys/smiles NumPy metadata pickle plus an exact-mass .npy array."""
    with Path(metadata_path).open("rb") as handle:
        metadata = _RestrictedNumpyUnpickler(handle).load()
    if not isinstance(metadata, dict) or not {"keys", "smiles"} <= set(metadata):
        raise ValueError("Metadata pickle must be a dict containing keys and smiles arrays")
    keys = np.asarray(metadata["keys"])
    smiles = np.asarray(metadata["smiles"])
    masses = np.load(mass_path, mmap_mode="r", allow_pickle=False)
    if not (keys.ndim == smiles.ndim == masses.ndim == 1):
        raise ValueError("Candidate arrays must be one-dimensional")
    if not (len(keys) == len(smiles) == len(masses)):
        raise ValueError("Candidate key, SMILES, and mass arrays have different lengths")
    frame = (
        pl.DataFrame(
            {
                "candidate_id": keys.astype(str),
                "smiles": smiles.astype(str),
                "canonical_smiles": smiles.astype(str),
                "inchikey14": [str(key)[:14] for key in keys],
                "formula": [None] * len(keys),
                "exact_mass": np.asarray(masses, dtype=np.float64),
                "source": [source] * len(keys),
            }
        )
        .filter(
            (pl.col("inchikey14").str.len_chars() == 14)
            & (pl.col("smiles").str.len_chars() > 0)
            & pl.col("exact_mass").is_finite()
            & (pl.col("exact_mass") > 0)
        )
        .unique(subset=["inchikey14"], keep="first", maintain_order=False)
    )
    database = CandidateDatabase(frame)
    database.save(output_path)
    return {
        "input_rows": len(keys),
        "output_rows": len(database),
        "duplicates_or_invalid": len(keys) - len(database),
    }


def merge_candidate_databases(
    paths: Iterable[str | Path], output_path: str | Path
) -> dict[str, int]:
    frames = [pl.read_parquet(path, columns=list(REQUIRED_COLUMNS)) for path in paths]
    if not frames:
        raise ValueError("At least one candidate database is required")
    input_rows = sum(len(frame) for frame in frames)
    merged = pl.concat(frames, how="vertical").unique(
        subset=["inchikey14"], keep="first", maintain_order=False
    )
    database = CandidateDatabase(merged)
    database.save(output_path)
    return {
        "input_rows": input_rows,
        "output_rows": len(database),
        "duplicates_removed": input_rows - len(database),
    }


def normalize_candidate_isotopes(
    input_path: str | Path, output_path: str | Path
) -> dict[str, int]:
    """Remove explicit isotope labels so neutral masses match ordinary sample molecules."""
    frame = pl.read_parquet(input_path, columns=list(REQUIRED_COLUMNS))
    input_rows = len(frame)
    isotope_rows = frame.filter(pl.col("smiles").str.contains(r"\[\d+[A-Z]"))
    updates: list[dict[str, object]] = []
    invalid = 0
    for row in isotope_rows.iter_rows(named=True):
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            invalid += 1
            continue
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        updates.append(
            {
                "inchikey14_old": row["inchikey14"],
                "smiles_new": smiles,
                "inchikey14_new": smiles_to_inchikey14(smiles),
                "formula_new": rdMolDescriptors.CalcMolFormula(mol),
                "exact_mass_new": float(rdMolDescriptors.CalcExactMolWt(mol)),
            }
        )
    if updates:
        update_frame = pl.DataFrame(updates)
        frame = (
            frame.join(
                update_frame,
                left_on="inchikey14",
                right_on="inchikey14_old",
                how="left",
            )
            .with_columns(
                pl.coalesce("smiles_new", "smiles").alias("smiles"),
                pl.coalesce("smiles_new", "canonical_smiles").alias("canonical_smiles"),
                pl.coalesce("inchikey14_new", "inchikey14").alias("inchikey14"),
                pl.coalesce("formula_new", "formula").alias("formula"),
                pl.coalesce("exact_mass_new", "exact_mass").alias("exact_mass"),
            )
            .select(REQUIRED_COLUMNS)
            .unique(subset=["inchikey14"], keep="first", maintain_order=True)
        )
    database = CandidateDatabase(frame)
    database.save(output_path)
    return {
        "input_rows": input_rows,
        "output_rows": len(database),
        "isotope_rows_found": len(isotope_rows),
        "isotope_rows_normalized": len(updates),
        "invalid_isotope_smiles": invalid,
    }
