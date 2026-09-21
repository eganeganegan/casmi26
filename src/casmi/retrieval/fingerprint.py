from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

from casmi.retrieval.candidate_db import CandidateDatabase


@dataclass(frozen=True, slots=True)
class FingerprintHit:
    inchikey14: str
    similarity: float


class CandidateFingerprintIndex:
    """Bit-packed Morgan fingerprints aligned to candidate InChIKey14 values."""

    def __init__(
        self,
        keys: np.ndarray,
        packed_fingerprints: np.ndarray,
        *,
        n_bits: int = 2048,
        radius: int = 2,
    ) -> None:
        self.keys = np.asarray(keys, dtype="S14")
        self.packed_fingerprints = np.asarray(packed_fingerprints, dtype=np.uint8)
        self.n_bits = int(n_bits)
        self.radius = int(radius)
        expected_bytes = (self.n_bits + 7) // 8
        if self.packed_fingerprints.shape != (len(self.keys), expected_bytes):
            raise ValueError(
                "packed_fingerprints must have shape "
                f"({len(self.keys)}, {expected_bytes}), got {self.packed_fingerprints.shape}"
            )
        if len(np.unique(self.keys)) != len(self.keys):
            raise ValueError("Fingerprint keys must be unique")
        self._key_to_index: dict[str, int] | None = None

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def storage_nbytes(self) -> int:
        return self.keys.nbytes + self.packed_fingerprints.nbytes

    @classmethod
    def from_candidate_database(
        cls,
        database: CandidateDatabase,
        *,
        n_bits: int = 2048,
        radius: int = 2,
    ) -> tuple[CandidateFingerprintIndex, dict[str, int]]:
        if n_bits <= 0:
            raise ValueError("n_bits must be positive")
        if radius < 0:
            raise ValueError("radius must be non-negative")
        key_array = np.empty(len(database), dtype="S14")
        packed_matrix = np.empty(
            (len(database), (n_bits + 7) // 8), dtype=np.uint8
        )
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        written = 0
        invalid = 0
        with rdBase.BlockLogs():
            for key, smiles in tqdm(
                database.frame.select("inchikey14", "smiles").iter_rows(),
                total=len(database),
                desc="candidate fingerprints",
                unit="structure",
            ):
                try:
                    mol = Chem.MolFromSmiles(str(smiles))
                    if mol is None:
                        raise ValueError("invalid SMILES")
                    fingerprint = generator.GetFingerprintAsNumPy(mol)
                except (ValueError, RuntimeError):
                    invalid += 1
                    continue
                key_array[written] = str(key)
                packed_matrix[written] = np.packbits(fingerprint, bitorder="little")
                written += 1
        index = cls(
            key_array[:written],
            packed_matrix[:written],
            n_bits=n_bits,
            radius=radius,
        )
        return index, {"input_structures": len(database), "indexed_structures": len(index), "invalid": invalid}

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> CandidateFingerprintIndex:
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(value).__name__}")
        return value

    def _ensure_lookup(self) -> dict[str, int]:
        if self._key_to_index is None:
            self._key_to_index = {
                value.decode("ascii"): index for index, value in enumerate(self.keys)
            }
        return self._key_to_index

    def indices_for_keys(self, keys: Iterable[str]) -> np.ndarray:
        lookup = self._ensure_lookup()
        return np.asarray(
            [lookup[key] for key in keys if key in lookup],
            dtype=np.int64,
        )

    def unpack(self, indices: np.ndarray | None = None) -> np.ndarray:
        packed = self.packed_fingerprints if indices is None else self.packed_fingerprints[indices]
        return np.unpackbits(packed, axis=1, count=self.n_bits, bitorder="little")

    def rank_probabilities(
        self,
        probabilities: np.ndarray,
        *,
        candidate_indices: np.ndarray | None = None,
        top_n: int = 100,
    ) -> list[FingerprintHit]:
        from casmi.models import fingerprint_compatibility

        if top_n < 0:
            raise ValueError("top_n must be non-negative")
        indices = (
            np.arange(len(self), dtype=np.int64)
            if candidate_indices is None
            else np.asarray(candidate_indices, dtype=np.int64)
        )
        if len(indices) == 0 or top_n == 0:
            return []
        fingerprints = self.unpack(indices)
        scores = fingerprint_compatibility(probabilities, fingerprints)["expected_tanimoto"]
        keep = min(top_n, len(indices))
        best_local = np.argpartition(scores, -keep)[-keep:]
        best_local = best_local[np.argsort(scores[best_local], kind="stable")[::-1]]
        return [
            FingerprintHit(
                inchikey14=self.keys[int(indices[local])].decode("ascii"),
                similarity=float(scores[local]),
            )
            for local in best_local
        ]
