"""Read-only validation and fingerprinting for the frozen aligned11 protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DATASET = "bci_subjects_ept_v6_aligned11"
FROZEN_DATA_ROOT = PROJECT_ROOT / "data" / "processed" / FROZEN_DATASET
FROZEN_MANIFESTS = {
    "train": f"data/processed/{FROZEN_DATASET}/manifests/sessions_all.jsonl",
    "val": f"data/processed/{FROZEN_DATASET}/manifests/sessions_val.jsonl",
    "test": f"data/processed/{FROZEN_DATASET}/manifests/sessions_test.jsonl",
}
FROZEN_COHORT_POLICY = "all_sessions_training"
FROZEN_SESSION_IDS = (
    "session_002",
    "session_003",
    "session_004",
    "session_006",
    "session_008",
    "session_009",
    "session_010",
    "session_012",
    "session_015",
    "session_017",
    "session_018",
)
FROZEN_EXCLUDED_IDS = ("session_011",)
METADATA_SUFFIXES = frozenset({".json", ".jsonl", ".yaml", ".yml"})

CONFIG_CONTRACTS = {
    "eptnet_marlin11_eeg_ppg_video.yaml": {
        "model": "eptnet",
        "modalities": {"eeg_time", "eeg_spectral", "ppg", "video"},
    },
    "lstr_marlin11_eeg_ppg_video.yaml": {
        "model": "lstr",
        "modalities": {"eeg_time", "eeg_spectral", "ppg", "video"},
    },
    "gatehub_marlin11_eeg_ppg_video.yaml": {
        "model": "gatehub",
        "modalities": {"eeg_time", "eeg_spectral", "ppg", "video"},
    },
    "testra_marlin11_eeg_ppg_video.yaml": {
        "model": "testra",
        "modalities": {"eeg_time", "eeg_spectral", "ppg", "video"},
    },
    "eptnet_marlin11_video_only.yaml": {
        "model": "eptnet",
        "modalities": {"video"},
    },
    "eptnet_marlin11_fixed_reader.yaml": {
        "model": "eptnet",
        "modalities": {"eeg_time", "eeg_spectral", "ppg", "video"},
        "adaptive_reader": False,
    },
    "eptnet_marlin11_no_persistent.yaml": {
        "model": "eptnet",
        "modalities": {"eeg_time", "eeg_spectral", "ppg", "video"},
        "persistent_state": False,
    },
}


def _load_config_module():
    """Load strict config validation without importing optional extractors."""
    source = PROJECT_ROOT / "src" / "eptnet" / "config.py"
    spec = importlib.util.spec_from_file_location("_eptnet_config", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load configuration module from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load_config = _load_config_module().load_config


def _load_provenance_module():
    """Load provenance helpers without importing optional model extractors."""

    source = PROJECT_ROOT / "src" / "eptnet" / "provenance.py"
    spec = importlib.util.spec_from_file_location("_eptnet_provenance", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load provenance module from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


provenance_module = _load_provenance_module()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _enabled_modalities(config: Mapping[str, Any]) -> set[str]:
    model = config["model"]
    enabled = set()
    for switch, name in (
        ("use_eeg_time", "eeg_time"),
        ("use_eeg_spec", "eeg_spectral"),
        ("use_hr", "ppg"),
    ):
        if model[switch]:
            enabled.add(name)
    if model["use_behavior_context"]:
        for switch, name in (
            ("use_video", "video"),
            ("use_audio", "audio"),
            ("use_text", "text"),
        ):
            if model[switch]:
                enabled.add(name)
    return enabled


def _manifest_session_ids(path: Path) -> list[str]:
    session_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"Manifest record is not an object: {path}:{line_number}")
            metadata = value.get("metadata", {})
            if not isinstance(metadata, Mapping):
                metadata = {}
            session_id = str(metadata.get("session_id", value.get("sample_id", "")))
            if not session_id:
                raise ValueError(f"Missing session ID: {path}:{line_number}")
            session_ids.append(session_id)
    if not session_ids:
        raise ValueError(f"Manifest is empty: {path}")
    if len(session_ids) != len(set(session_ids)):
        raise ValueError(f"Manifest contains duplicate session IDs: {path}")
    return session_ids


def _manifest_tensor_fingerprints(
    path: Path,
    *,
    dataset_root: Path,
    known_fingerprints: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Verify every frozen session tensor against its manifest checksum."""

    root = dataset_root.resolve()
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"Manifest record is not an object: {path}:{line_number}")
            tensor_file = value.get("tensor_file")
            expected_sha256 = value.get("sha256")
            if not isinstance(tensor_file, str) or not tensor_file:
                raise ValueError(f"Missing tensor_file: {path}:{line_number}")
            if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
                raise ValueError(f"Invalid tensor sha256: {path}:{line_number}")
            tensor_path = (path.parent / tensor_file).resolve()
            try:
                relative = tensor_path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"Tensor path escapes frozen dataset: {path}:{line_number}"
                ) from error
            if not tensor_path.is_file():
                raise FileNotFoundError(f"Frozen session tensor is missing: {tensor_path}")
            known = (
                known_fingerprints.get(relative.as_posix())
                if known_fingerprints is not None
                else None
            )
            if known_fingerprints is not None and not isinstance(known, Mapping):
                raise ValueError(
                    f"Runtime provenance omitted frozen tensor: {relative.as_posix()}"
                )
            actual_sha256 = (
                str(known["sha256"]) if isinstance(known, Mapping) else _sha256_file(tensor_path)
            )
            actual_bytes = (
                int(known["bytes"])
                if isinstance(known, Mapping)
                else tensor_path.stat().st_size
            )
            if tensor_path.stat().st_size != actual_bytes:
                raise RuntimeError(
                    f"Frozen tensor size changed during preflight: {relative.as_posix()}"
                )
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Frozen tensor checksum mismatch: {relative.as_posix()}"
                )
            records.append(
                {
                    "path": relative.as_posix(),
                    "bytes": actual_bytes,
                    "sha256": actual_sha256,
                }
            )
    return records


def _metadata_fingerprints(dataset_root: Path) -> tuple[list[dict[str, Any]], str]:
    records = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in METADATA_SUFFIXES:
            continue
        relative = path.relative_to(dataset_root).as_posix()
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return records, digest.hexdigest()


def _resolve_device(value: str) -> str:
    import torch

    if value == "auto":
        if torch.cuda.is_available():
            return str(torch.device("cuda", torch.cuda.current_device()))
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({value}) but CUDA is not available")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is invalid; detected "
                f"{torch.cuda.device_count()} device(s)"
            )
        return str(torch.device("cuda", index))
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS device requested but MPS is not available")
    return str(device)


def validate_configs(
    paths: list[Path], *, allow_missing_data: bool, device: str | None
) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one paper-aligned configuration is required")

    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    output_directories: set[str] = set()
    for supplied in paths:
        path = supplied if supplied.is_absolute() else PROJECT_ROOT / supplied
        if not path.is_file():
            raise FileNotFoundError(f"Configuration does not exist: {path}")
        contract = CONFIG_CONTRACTS.get(path.name)
        if contract is None:
            raise ValueError(f"No frozen paper contract for {path.name}")
        config = load_config(str(path))
        data = config["data"]
        model = config["model"]
        evaluation = config["evaluation"]
        experiment = config["experiment"]

        if data["dataset_name"] != FROZEN_DATASET:
            raise ValueError(f"{path.name} changes the frozen dataset identity")
        actual_manifests = {
            split: data[f"{split}_manifest"] for split in ("train", "val", "test")
        }
        if actual_manifests != FROZEN_MANIFESTS:
            raise ValueError(f"{path.name} changes one or more frozen manifest paths")
        if data.get("cohort_policy") != FROZEN_COHORT_POLICY:
            raise ValueError(f"{path.name} changes the all-session training policy")
        if tuple(data.get("expected_session_ids", ())) != FROZEN_SESSION_IDS:
            raise ValueError(f"{path.name} changes the frozen aligned11 cohort")
        if tuple(data.get("excluded_session_ids", ())) != FROZEN_EXCLUDED_IDS:
            raise ValueError(f"{path.name} changes the frozen exclusion contract")
        if data.get("require_aligned_behavior_modalities") is not True:
            raise ValueError(f"{path.name} disables the frozen alignment gate")
        if data["num_classes"] != 2 or evaluation["positive_class"] != 0:
            raise ValueError(f"{path.name} changes the immutable label/event mapping")
        if model["use_audio"] or model["use_text"]:
            raise ValueError(f"{path.name} must disable audio and text")
        if model["name"] != contract["model"]:
            raise ValueError(f"{path.name} uses the wrong model family")
        modalities = _enabled_modalities(config)
        if modalities != contract["modalities"]:
            raise ValueError(
                f"{path.name} enables {sorted(modalities)}; "
                f"expected {sorted(contract['modalities'])}"
            )
        for key in ("adaptive_reader", "persistent_state"):
            if key in contract and model[key] is not contract[key]:
                raise ValueError(f"{path.name} violates the {key} diagnostic contract")
        output_dir = experiment["output_dir"]
        if not output_dir.startswith(
            (
                "results/eptnet_marlin11_",
                "results/lstr_marlin11_",
                "results/gatehub_marlin11_",
                "results/testra_marlin11_",
            )
        ) or not output_dir.endswith("_no_text"):
            raise ValueError(f"{path.name} does not use an isolated paper result directory")
        if output_dir in output_directories:
            raise ValueError(f"Duplicate paper result directory: {output_dir}")
        output_directories.add(output_dir)
        records.append(
            {
                "config": path.relative_to(PROJECT_ROOT).as_posix(),
                "experiment": experiment["name"],
                "output_dir": output_dir,
                "model": model["name"],
                "modalities": sorted(modalities),
                "training_config_sha256": provenance_module.training_config_fingerprint(
                    config
                ),
            }
        )

    strict_provenance = None
    known_tensor_fingerprints: dict[str, Mapping[str, Any]] = {}
    if not allow_missing_data:
        if not FROZEN_DATA_ROOT.is_dir():
            raise FileNotFoundError(f"Frozen dataset root is missing: {FROZEN_DATA_ROOT}")
        strict_provenance = provenance_module.build_provenance_fingerprint(
            PROJECT_ROOT,
            FROZEN_DATA_ROOT,
            manifest_paths=[
                _resolve_project_path(FROZEN_MANIFESTS[split])
                for split in ("train", "val", "test")
            ],
            requirements_lock_path=PROJECT_ROOT / "requirements-lock.txt",
            include_git=True,
        )
        for manifest in strict_provenance["manifests"]:
            for tensor in manifest["tensor_files"]:
                known_tensor_fingerprints[str(tensor["path"])] = tensor

    manifests: dict[str, Any] = {}
    session_tensors_by_path: dict[str, dict[str, Any]] = {}
    observed_ids: dict[str, list[str]] = {}
    data_complete = True
    for split, relative in FROZEN_MANIFESTS.items():
        path = _resolve_project_path(relative)
        if allow_missing_data:
            present = path.is_file()
            manifests[split] = {"path": relative, "present": present}
            data_complete = False
            if not present:
                warnings.append(f"Frozen {split} manifest is missing: {path}")
            continue
        if not path.is_file():
            data_complete = False
            message = f"Frozen {split} manifest is missing: {path}"
            raise FileNotFoundError(message)
        ids = _manifest_session_ids(path)
        tensor_fingerprints = _manifest_tensor_fingerprints(
            path,
            dataset_root=FROZEN_DATA_ROOT,
            known_fingerprints=known_tensor_fingerprints,
        )
        observed_ids[split] = ids
        for record in tensor_fingerprints:
            session_tensors_by_path[str(record["path"])] = record
        manifests[split] = {
            "path": relative,
            "present": True,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "session_ids": ids,
        }
    if data_complete:
        train_ids = set(observed_ids["train"])
        if train_ids != set(FROZEN_SESSION_IDS):
            raise ValueError(
                "Frozen all-session training manifest mismatch; "
                f"missing={sorted(set(FROZEN_SESSION_IDS) - train_ids)}, "
                f"unexpected={sorted(train_ids - set(FROZEN_SESSION_IDS))}"
            )
        evaluation_ids = set(observed_ids["val"]).union(observed_ids["test"])
        if not evaluation_ids.issubset(train_ids):
            raise ValueError("Evaluation manifests escape the all-session training cohort")
        if set(observed_ids["val"]).intersection(observed_ids["test"]):
            raise ValueError("Frozen validation and test evaluation views overlap")
        if train_ids.intersection(FROZEN_EXCLUDED_IDS):
            raise ValueError("An explicitly excluded session occurs in the frozen manifests")

    metadata_files: list[dict[str, Any]] = []
    metadata_sha256 = None
    summary_contract = None
    runtime_provenance = None
    if allow_missing_data and FROZEN_DATA_ROOT.is_dir():
        warnings.append(
            "Strict frozen-data hashing is skipped in allow-missing-data mode"
        )
    elif FROZEN_DATA_ROOT.is_dir():
        metadata_files, metadata_sha256 = _metadata_fingerprints(FROZEN_DATA_ROOT)
        if not metadata_files:
            raise ValueError(f"No frozen metadata files found below {FROZEN_DATA_ROOT}")
        if data_complete:
            success_path = FROZEN_DATA_ROOT / "_SUCCESS.json"
            summary_path = FROZEN_DATA_ROOT / "dataset_summary.json"
            if not success_path.is_file():
                raise FileNotFoundError(f"Frozen success marker is missing: {success_path}")
            if not summary_path.is_file():
                raise FileNotFoundError(f"Frozen dataset summary is missing: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary, Mapping):
                raise ValueError(f"Frozen dataset summary must be an object: {summary_path}")
            if summary.get("dataset") != FROZEN_DATASET:
                raise ValueError("Frozen dataset summary has the wrong dataset identity")
            if int(summary.get("num_subjects", -1)) != len(FROZEN_SESSION_IDS):
                raise ValueError("Frozen dataset summary must contain exactly 11 subjects")
            split_sessions = summary.get("split_sessions")
            if not isinstance(split_sessions, Mapping):
                raise ValueError("Frozen dataset summary is missing split_sessions")
            normalized_splits = {
                split: [str(value) for value in split_sessions.get(split, [])]
                for split in ("train", "val", "test")
            }
            for split in ("val", "test"):
                values = normalized_splits[split]
                if values != manifests[split]["session_ids"]:
                    raise ValueError(
                        f"dataset_summary split_sessions disagrees with {split} manifest"
                    )
            artifact_ids = {
                value for values in normalized_splits.values() for value in values
            }
            if artifact_ids != set(FROZEN_SESSION_IDS):
                raise ValueError("dataset_summary does not cover the frozen 11-session cohort")
            summary_contract = {
                "dataset": FROZEN_DATASET,
                "num_subjects": len(FROZEN_SESSION_IDS),
                "artifact_split_sessions": normalized_splits,
                "training_cohort_policy": FROZEN_COHORT_POLICY,
                "training_session_ids": manifests["train"]["session_ids"],
                "evaluation_views": {
                    split: manifests[split]["session_ids"] for split in ("val", "test")
                },
            }
        if data_complete:
            if strict_provenance is None:
                raise RuntimeError("Strict runtime provenance was not computed")
            provenance = strict_provenance
            runtime_provenance = {
                "schema_version": provenance["schema_version"],
                "provenance_sha256": provenance["provenance_sha256"],
                "source_tree_sha256": provenance["source_tree"]["sha256"],
                "required_artifacts": {
                    name: record["sha256"]
                    for name, record in sorted(provenance["required_artifacts"].items())
                },
                "manifests": {
                    record["path"]: record["manifest_and_tensors_sha256"]
                    for record in provenance["manifests"]
                },
            }
    else:
        message = f"Frozen dataset root is missing: {FROZEN_DATA_ROOT}"
        if allow_missing_data:
            warnings.append(message)
        else:
            raise FileNotFoundError(message)

    return {
        "status": "passed_with_warnings" if warnings else "passed",
        "mode": "read_only_frozen_evidence_check",
        "dataset": FROZEN_DATASET,
        "label_mapping": {"0": "deception", "1": "truth"},
        "positive_class": 0,
        "cohort_policy": FROZEN_COHORT_POLICY,
        "expected_session_ids": list(FROZEN_SESSION_IDS),
        "excluded_session_ids": list(FROZEN_EXCLUDED_IDS),
        "resolved_device": None if device is None else _resolve_device(device),
        "configs": records,
        "manifests": manifests,
        "session_tensors": [
            session_tensors_by_path[path] for path in sorted(session_tensors_by_path)
        ],
        "dataset_summary_contract": summary_contract,
        "runtime_provenance": runtime_provenance,
        "metadata_files": metadata_files,
        "metadata_bundle_sha256": metadata_sha256,
        "warnings": warnings,
    }


def assert_frozen_evidence_matches(
    reference: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Reject a resumed run if any frozen input fingerprint has changed."""
    evidence_keys = (
        "dataset",
        "label_mapping",
        "positive_class",
        "cohort_policy",
        "expected_session_ids",
        "excluded_session_ids",
        "manifests",
        "session_tensors",
        "dataset_summary_contract",
        "runtime_provenance",
        "metadata_files",
        "metadata_bundle_sha256",
    )
    changed = [key for key in evidence_keys if reference.get(key) != current.get(key)]
    if changed:
        raise ValueError(
            "Frozen aligned11 evidence differs from the recorded preflight: "
            + ", ".join(changed)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-device-check", action="store_true")
    parser.add_argument("--allow-missing-data", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Existing preflight JSON whose frozen evidence fingerprints must match",
    )
    args = parser.parse_args()

    payload = validate_configs(
        args.configs,
        allow_missing_data=args.allow_missing_data,
        device=None if args.skip_device_check else args.device,
    )
    if args.reference is not None:
        reference_path = (
            args.reference if args.reference.is_absolute() else PROJECT_ROOT / args.reference
        )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if not isinstance(reference, Mapping):
            raise ValueError(f"Preflight reference must contain a JSON object: {reference_path}")
        assert_frozen_evidence_matches(reference, payload)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to overwrite preflight record: {output}"
            ) from error
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
