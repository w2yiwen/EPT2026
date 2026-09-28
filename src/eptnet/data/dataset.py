from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .schema import (
    COMMON_INPUTS,
    OPTIONAL_AVAILABILITY,
    OPTIONAL_DECISION_MASKS,
    OPTIONAL_TARGETS,
    PRECOMPUTED_INPUTS,
    RAW_WINDOW_INPUTS,
    validate_batch,
)


class ManifestDataset(Dataset):
    """Dataset adapter for JSONL manifests pointing to per-session ``.pt`` files.

    Each line must contain ``sample_id`` and ``tensor_file``. The tensor file is a
    dictionary without a batch axis, using the keys documented in ``schema.py``.
    This generic adapter is intentionally the only data-specific seam in the scaffold.
    """

    def __init__(
        self,
        manifest_path: str,
        require_targets: bool = True,
        verify_checksums: bool = True,
    ) -> None:
        if not manifest_path:
            raise ValueError("manifest_path must point to a prepared JSONL manifest")
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        self.require_targets = require_targets
        self.verify_checksums = verify_checksums
        self._verified_paths: set[Path] = set()
        self.records: list[dict[str, Any]] = []
        sample_ids: set[str] = set()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Manifest line {line_number} must be a JSON object")
                if "sample_id" not in record or "tensor_file" not in record:
                    raise ValueError(f"Invalid manifest line {line_number}: {record}")
                sample_id = str(record["sample_id"])
                if sample_id in sample_ids:
                    raise ValueError(f"Duplicate sample_id {sample_id!r} at line {line_number}")
                sample_ids.add(sample_id)
                self.records.append(record)
        if not self.records:
            raise ValueError(f"No samples found in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._load_record(self.records[index])

    def _load_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        tensor_path = Path(record["tensor_file"])
        if not tensor_path.is_absolute():
            tensor_path = self.manifest_path.parent / tensor_path
        tensor_path = tensor_path.resolve()
        dataset_root = self.manifest_path.parent.parent.resolve()
        try:
            tensor_path.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError(
                f"Tensor path escapes dataset root {dataset_root}: {tensor_path}"
            ) from error
        if tensor_path.suffix.lower() != ".pt" or not tensor_path.is_file():
            raise FileNotFoundError(f"Expected a prepared .pt tensor file: {tensor_path}")
        expected_hash = record.get("sha256")
        if self.verify_checksums and expected_hash is not None and tensor_path not in self._verified_paths:
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError(f"Invalid sha256 value for {tensor_path}")
            expected_hash = expected_hash.lower()
            if any(character not in "0123456789abcdef" for character in expected_hash):
                raise ValueError(f"Invalid sha256 value for {tensor_path}")
            digest = hashlib.sha256()
            with tensor_path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f"Checksum mismatch for prepared tensor: {tensor_path}")
            self._verified_paths.add(tensor_path)
        # Prepared windows contain tensors and primitive Python containers only.
        # Keep the restricted unpickler enabled so a manifest cannot turn an
        # untrusted ``.pt`` file into arbitrary-code execution during loading.
        sample = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(sample, dict):
            raise TypeError(f"Expected a tensor dictionary in {tensor_path}")
        if self.require_targets:
            missing_targets = [key for key in OPTIONAL_TARGETS if key not in sample]
            if missing_targets:
                raise ValueError(f"Missing targets {missing_targets} in {tensor_path}")
        sample["sample_id"] = record["sample_id"]
        sample["metadata"] = record.get("metadata", {})
        return sample


class StitchedManifestDataset(Dataset):
    """Reconstruct one unique, continuous sequence per session from windows.

    Prepared manifests may contain overlapping windows.  Treating those windows as
    independent evaluation samples double-counts rows and fragments events at window
    edges.  This adapter uses each window's explicit ``row_indices`` as the source of
    truth, verifies every duplicated temporal field exactly, and returns rows once in
    source order.
    """

    def __init__(self, manifest_path: str, require_targets: bool = True) -> None:
        self.windows = ManifestDataset(manifest_path, require_targets=require_targets)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for line_number, record in enumerate(self.windows.records, 1):
            metadata = record.get("metadata", {})
            session_id = metadata.get("session_id")
            if session_id is None or not str(session_id).strip():
                raise ValueError(
                    f"Manifest line {line_number} needs metadata.session_id for stitching"
                )
            grouped.setdefault(str(session_id), []).append(record)
        self._session_records = [(key, grouped[key]) for key in sorted(grouped)]
        self._materialized: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._session_records)

    def materialize(self) -> None:
        """Load every session once so a run cannot observe mid-run file replacement."""

        for index in range(len(self)):
            self[index]

    @staticmethod
    def _temporal_fields(
        sample: Mapping[str, Any], length: int
    ) -> tuple[dict[str, Tensor], dict[str, Sequence[Any]]]:
        tensors: dict[str, Tensor] = {}
        sequences: dict[str, Sequence[Any]] = {}
        for key, value in sample.items():
            if key in {"row_indices", "sample_id", "metadata"}:
                continue
            if isinstance(value, Tensor):
                if value.ndim == 0 or value.shape[0] != length:
                    raise ValueError(
                        f"Temporal tensor {key!r} has shape {tuple(value.shape)}; "
                        f"expected leading length {length}"
                    )
                tensors[key] = value
            elif isinstance(value, (list, tuple)):
                if len(value) != length:
                    raise ValueError(
                        f"Temporal sequence {key!r} has length {len(value)}; expected {length}"
                    )
                sequences[key] = value
        return tensors, sequences

    @staticmethod
    def _same_tensor(first: Tensor, second: Tensor) -> bool:
        return (
            first.dtype == second.dtype
            and first.shape == second.shape
            and torch.equal(first, second)
        )

    def _stitch(self, session_id: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rows: dict[int, dict[str, dict[str, Any]]] = {}
        tensor_keys: set[str] | None = None
        sequence_keys: set[str] | None = None
        row_dtype: torch.dtype | None = None
        metadata_values: dict[str, set[Any]] = {"dataset": set(), "split": set()}

        for record in records:
            sample = self.windows._load_record(record)
            if "row_indices" not in sample or not isinstance(sample["row_indices"], Tensor):
                raise ValueError(f"Sample {record['sample_id']} is missing tensor row_indices")
            row_indices = sample["row_indices"]
            if row_indices.ndim != 1:
                raise ValueError(
                    f"row_indices in {record['sample_id']} must be rank 1, "
                    f"got {tuple(row_indices.shape)}"
                )
            indices = [int(value) for value in row_indices.tolist()]
            if len(indices) != len(set(indices)):
                raise ValueError(f"Duplicate row_indices inside sample {record['sample_id']}")
            row_dtype = row_dtype or row_indices.dtype
            tensors, sequences = self._temporal_fields(sample, len(indices))
            current_tensor_keys = set(tensors)
            current_sequence_keys = set(sequences)
            if tensor_keys is None:
                tensor_keys = current_tensor_keys
                sequence_keys = current_sequence_keys
            elif current_tensor_keys != tensor_keys or current_sequence_keys != sequence_keys:
                raise ValueError(
                    f"Temporal fields differ across windows for session {session_id}: "
                    f"tensor fields {sorted(current_tensor_keys)} vs {sorted(tensor_keys)}, "
                    f"sequence fields {sorted(current_sequence_keys)} vs "
                    f"{sorted(sequence_keys or set())}"
                )

            metadata = record.get("metadata", {})
            for key in metadata_values:
                if key in metadata:
                    metadata_values[key].add(metadata[key])

            for offset, row_index in enumerate(indices):
                tensor_values = {key: value[offset] for key, value in tensors.items()}
                sequence_values = {key: value[offset] for key, value in sequences.items()}
                if row_index not in rows:
                    rows[row_index] = {
                        "tensors": {key: value.clone() for key, value in tensor_values.items()},
                        "sequences": dict(sequence_values),
                    }
                    continue
                existing = rows[row_index]
                for key, value in tensor_values.items():
                    if not self._same_tensor(existing["tensors"][key], value):
                        raise ValueError(
                            f"Inconsistent duplicate row {row_index} field {key!r} "
                            f"in session {session_id}"
                        )
                for key, value in sequence_values.items():
                    if existing["sequences"][key] != value:
                        raise ValueError(
                            f"Inconsistent duplicate row {row_index} field {key!r} "
                            f"in session {session_id}"
                        )

        ordered_rows = sorted(rows)
        if not ordered_rows:
            raise ValueError(f"No rows found for session {session_id}")
        for previous, current in zip(ordered_rows, ordered_rows[1:], strict=False):
            if current != previous + 1:
                raise ValueError(
                    f"Non-contiguous rows for session {session_id}: "
                    f"expected {previous + 1}, found {current}"
                )

        output: dict[str, Any] = {
            "row_indices": torch.tensor(ordered_rows, dtype=row_dtype or torch.long),
            "sample_id": f"{session_id}_stitched",
        }
        for key in sorted(tensor_keys or set()):
            output[key] = torch.stack([rows[index]["tensors"][key] for index in ordered_rows])
        for key in sorted(sequence_keys or set()):
            output[key] = [rows[index]["sequences"][key] for index in ordered_rows]

        output_metadata: dict[str, Any] = {
            "session_id": session_id,
            "source_row_start": ordered_rows[0],
            "source_row_end_exclusive": ordered_rows[-1] + 1,
            "source_window_count": len(records),
        }
        for key, values in metadata_values.items():
            if len(values) == 1:
                output_metadata[key] = next(iter(values))
            elif values:
                output_metadata[f"{key}s"] = sorted(values)
        output["metadata"] = output_metadata
        return output

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self._session_records)
        if index < 0 or index >= len(self._session_records):
            raise IndexError(index)
        if index in self._materialized:
            return self._materialized[index]
        session_id, records = self._session_records[index]
        sample = self._stitch(session_id, records)
        self._materialized[index] = sample
        return sample


def collate_multimodal(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length sessions and add the batch axis."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    tensor_keys = (
        set(COMMON_INPUTS)
        | set(RAW_WINDOW_INPUTS)
        | set(PRECOMPUTED_INPUTS)
        | set(OPTIONAL_AVAILABILITY)
        | set(OPTIONAL_DECISION_MASKS)
        | set(OPTIONAL_TARGETS)
        | {"timestamps", "row_indices"}
    )
    tensor_keys.remove("sequence_mask")
    output: dict[str, Any] = {}
    for key in tensor_keys:
        values = [sample[key] for sample in samples if key in sample]
        if not values:
            continue
        if len(values) != len(samples):
            raise ValueError(f"Target/input key {key} is present for only part of the batch")
        padding_value = -1 if key == "row_indices" else 0
        output[key] = pad_sequence(values, batch_first=True, padding_value=padding_value)

    length_key = "eeg_time" if "eeg_time" in samples[0] else "eeg"
    lengths = torch.tensor([sample[length_key].shape[0] for sample in samples])
    max_steps = int(lengths.max().item())
    output["sequence_mask"] = torch.arange(max_steps)[None, :] < lengths[:, None]
    output["sample_id"] = [sample["sample_id"] for sample in samples]
    output["metadata"] = [sample.get("metadata", {}) for sample in samples]
    validate_batch(output, require_targets="labels" in output)
    return output
