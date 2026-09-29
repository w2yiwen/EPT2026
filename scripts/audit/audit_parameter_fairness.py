"""Audit entry point for executed-model parameter fairness."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from eptnet.config import load_config
from eptnet.data import ManifestDataset, collate_multimodal
from eptnet.losses import EPTNetLoss
from eptnet.models import build_model
from eptnet.train import resolve_device

DEFAULT_CONFIGS = (
    "configs/default.yaml",
    "configs/baseline_early_fusion_gru.yaml",
    "configs/baseline_fusion_transformer.yaml",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_INPUT_SCHEMA_VERSION = 1
AUDIT_SCRIPT = Path(__file__).resolve()
IMPLEMENTATION_SOURCE_FILES = (
    PROJECT_ROOT / "src/eptnet/config.py",
    PROJECT_ROOT / "src/eptnet/data/__init__.py",
    PROJECT_ROOT / "src/eptnet/data/dataset.py",
    PROJECT_ROOT / "src/eptnet/data/schema.py",
    PROJECT_ROOT / "src/eptnet/losses.py",
    PROJECT_ROOT / "src/eptnet/train.py",
    AUDIT_SCRIPT,
)
IMPLEMENTATION_SOURCE_DIRECTORIES = (PROJECT_ROOT / "src/eptnet/models",)

MODALITY_ORDER = (
    "eeg_time",
    "eeg_spectral",
    "hr",
    "video",
    "audio",
    "text",
)
PHYSIOLOGY_MODALITIES = MODALITY_ORDER[:3]
BEHAVIOR_MODALITIES = MODALITY_ORDER[3:]
MODEL_SWITCHES = {
    "eeg_time": "use_eeg_time",
    "eeg_spectral": "use_eeg_spec",
    "hr": "use_hr",
    "video": "use_video",
    "audio": "use_audio",
    "text": "use_text",
}
GRAPH_COUNTING_PROTOCOL = (
    "trainable parameter elements whose parameter tensor has a non-None gradient after "
    "deterministic accumulated full-multitask backward passes over the shared minimal "
    "target-valid modality-cover sample set; graph participation does not imply a "
    "non-zero update"
)
NONZERO_PARAMETER_COUNTING_PROTOCOL = (
    "trainable parameter elements belonging to a parameter tensor with at least one "
    "finite exactly non-zero accumulated gradient element on the shared audit sample set"
)
NONZERO_ELEMENT_COUNTING_PROTOCOL = (
    "individual trainable parameter elements with a finite exactly non-zero accumulated "
    "gradient on the shared audit sample set"
)
SELECTION_PROTOCOL = (
    "minimum-cardinality set of training-manifest windows covering every model-enabled "
    "modality that is observed during at least one target-valid step; ties are resolved "
    "by the lexicographically smallest ordered sample-id tuple"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _portable_repository_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _portable_declared_path(value: str, *, field: str) -> str:
    """Render a declared repository path without resolving private symlink targets."""

    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            path = path.relative_to(PROJECT_ROOT)
        except ValueError as error:
            raise ValueError(
                f"{field} must be declared as a repository-relative path; "
                "external absolute paths are not recorded in audit artifacts"
            ) from error
    if ".." in path.parts:
        raise ValueError(f"{field} must not escape the repository")
    return path.as_posix()


def _config_dependency_paths(path: Path, active: tuple[Path, ...] = ()) -> tuple[Path, ...]:
    """Return the complete inheritance chain used by one YAML config."""

    canonical = path.expanduser().resolve()
    if canonical in active:
        chain = " -> ".join(item.as_posix() for item in (*active, canonical))
        raise ValueError(f"Cyclic config inheritance detected while hashing: {chain}")
    if not canonical.is_file():
        raise FileNotFoundError(canonical)
    loaded = yaml.safe_load(canonical.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Configuration at {canonical} must be a YAML mapping")
    inherited = loaded.get("inherits")
    if inherited is None:
        return (canonical,)
    if not isinstance(inherited, str) or not inherited.strip():
        raise ValueError(f"inherits in {canonical} must be a non-empty path string")
    parent = canonical.parent / inherited
    dependencies = _config_dependency_paths(parent, (*active, canonical))
    return (*dependencies, canonical)


def _implementation_source_paths() -> tuple[Path, ...]:
    candidates = list(IMPLEMENTATION_SOURCE_FILES)
    for directory in IMPLEMENTATION_SOURCE_DIRECTORIES:
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Fairness-audit implementation directory is missing: "
                f"{_portable_repository_path(directory)}"
            )
        candidates.extend(directory.rglob("*.py"))
    paths = tuple(
        sorted({path.resolve() for path in candidates}, key=_portable_repository_path)
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        rendered = ", ".join(_portable_repository_path(path) for path in missing)
        raise FileNotFoundError(f"Fairness-audit implementation source is missing: {rendered}")
    return paths


def _manifest_identity(configs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not configs:
        raise ValueError("At least one configuration is required")
    declared_paths = [str(config["data"]["train_manifest"]) for config in configs]
    portable_paths = [
        _portable_declared_path(path, field="data.train_manifest") for path in declared_paths
    ]
    if any(path != portable_paths[0] for path in portable_paths[1:]):
        raise ValueError(
            "All compared configurations must declare the same portable train manifest"
        )
    resolved_paths = [Path(path).expanduser().resolve() for path in declared_paths]
    if any(path != resolved_paths[0] for path in resolved_paths[1:]):
        raise ValueError("All compared configurations must resolve to one train manifest")
    manifest_path = resolved_paths[0]
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return {
        "path": portable_paths[0],
        "sha256": _sha256_file(manifest_path),
    }


def _build_audit_inputs(
    config_paths: Sequence[str], configs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(config_paths) != len(configs):
        raise ValueError("Configuration paths and resolved configurations must align")
    config_records: list[dict[str, Any]] = []
    for config_path, config in zip(config_paths, configs, strict=True):
        portable_config_path = _portable_declared_path(config_path, field="config path")
        path = Path(config_path).expanduser().resolve()
        dependencies = _config_dependency_paths(path)
        try:
            path.relative_to(PROJECT_ROOT)
            for dependency in dependencies:
                dependency.relative_to(PROJECT_ROOT)
        except ValueError as error:
            raise ValueError(
                "Config files and their inheritance dependencies must remain inside "
                "the repository"
            ) from error
        dependency_hashes = {
            _portable_repository_path(dependency): _sha256_file(dependency)
            for dependency in dependencies
        }
        config_records.append(
            {
                "path": portable_config_path,
                "file_sha256": _sha256_file(path),
                "dependency_file_sha256": dependency_hashes,
                "resolved_config_sha256": _sha256_json(config),
            }
        )

    source_hashes = {
        _portable_repository_path(path): _sha256_file(path)
        for path in _implementation_source_paths()
    }
    payload: dict[str, Any] = {
        "schema_version": AUDIT_INPUT_SCHEMA_VERSION,
        "configs": config_records,
        "implementation_source_sha256": source_hashes,
        "implementation_source_bundle_sha256": _sha256_json(source_hashes),
        "train_manifest": _manifest_identity(configs),
    }
    return {**payload, "bundle_sha256": _sha256_json(payload)}


def _validate_identity_bundle(identity: Mapping[str, Any]) -> None:
    stored = identity.get("bundle_sha256")
    if not isinstance(stored, str) or len(stored) != 64:
        raise ValueError("Parameter-fairness reference lacks a valid audit-input bundle hash")
    payload = {key: value for key, value in identity.items() if key != "bundle_sha256"}
    if _sha256_json(payload) != stored:
        raise ValueError("Parameter-fairness reference audit-input bundle is internally invalid")


def assert_audit_reference_matches(
    reference: Mapping[str, Any], current_inputs: Mapping[str, Any]
) -> None:
    """Fail closed unless an existing audit was built from the exact current inputs."""

    reference_inputs = reference.get("audit_inputs")
    if not isinstance(reference_inputs, Mapping):
        raise ValueError(
            "Parameter-fairness reference has no audit_inputs fingerprint; "
            "legacy or incomplete reports cannot be reused"
        )
    _validate_identity_bundle(reference_inputs)
    _validate_identity_bundle(current_inputs)
    if reference_inputs != current_inputs:
        keys = sorted(
            key
            for key in set(reference_inputs).union(current_inputs)
            if reference_inputs.get(key) != current_inputs.get(key)
        )
        raise ValueError(
            "Parameter-fairness reference does not match current inputs: " + ", ".join(keys)
        )

    manifest = reference_inputs.get("train_manifest")
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("sha256"), str):
        raise ValueError("Parameter-fairness reference lacks a valid manifest fingerprint")
    selection = reference.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Parameter-fairness reference lacks its sample selection")
    if selection.get("manifest_sha256") != manifest["sha256"]:
        raise ValueError(
            "Parameter-fairness reference manifest hashes disagree between inputs and selection"
        )
    records = reference.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Parameter-fairness reference has no audit records")
    config_records = reference_inputs.get("configs")
    if not isinstance(config_records, list) or len(records) != len(config_records):
        raise ValueError(
            "Parameter-fairness reference record count does not match its config fingerprints"
        )
    expected_config_paths = [record.get("path") for record in config_records]
    actual_config_paths = [
        record.get("config") if isinstance(record, Mapping) else None for record in records
    ]
    if actual_config_paths != expected_config_paths:
        raise ValueError(
            "Parameter-fairness reference record configs do not match its config fingerprints"
        )
    if any(
        not isinstance(record, Mapping)
        or record.get("manifest_sha256") != manifest["sha256"]
        for record in records
    ):
        raise ValueError("Parameter-fairness record manifest hashes are inconsistent")


def check_audit_reference(config_paths: Sequence[str], reference_path: Path) -> dict[str, Any]:
    """Read and validate an audit reference without loading tensors or writing files."""

    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if not isinstance(reference, Mapping):
        raise ValueError("Parameter-fairness reference must be a JSON object")
    configs = [load_config(path) for path in config_paths]
    current_inputs = _build_audit_inputs(config_paths, configs)
    assert_audit_reference_matches(reference, current_inputs)
    return {
        "status": "matched",
        "reference": reference_path.as_posix(),
        "audit_input_bundle_sha256": current_inputs["bundle_sha256"],
        "manifest_sha256": current_inputs["train_manifest"]["sha256"],
    }


def _enabled_modalities(config: Mapping[str, Any]) -> tuple[str, ...]:
    model = config.get("model", config)
    behavior_enabled = bool(model.get("use_behavior_context", True))
    enabled: list[str] = []
    for modality in MODALITY_ORDER:
        if modality in BEHAVIOR_MODALITIES and not behavior_enabled:
            continue
        if bool(model.get(MODEL_SWITCHES[modality], modality != "text")):
            enabled.append(modality)
    return tuple(enabled)


def _target_valid_mask(sample: Mapping[str, Any]) -> Tensor:
    if "target_mask" in sample:
        mask = sample["target_mask"]
    elif "labels" in sample:
        mask = torch.ones_like(sample["labels"], dtype=torch.bool)
    elif "eeg_time" in sample:
        mask = torch.ones(sample["eeg_time"].shape[0], dtype=torch.bool)
    elif "eeg" in sample:
        mask = torch.ones(sample["eeg"].shape[0], dtype=torch.bool)
    else:
        raise ValueError(f"Cannot infer sequence length for sample {sample.get('sample_id')!r}")
    if not isinstance(mask, Tensor) or mask.ndim != 1:
        raise ValueError(f"target_mask must be rank 1 in sample {sample.get('sample_id')!r}")
    # Overlapping preparation windows can legitimately fall entirely inside an
    # interviewer span.  They cannot be selected for the audit cover, but their
    # presence in the manifest is not itself a contract violation.
    return mask.bool()


def _availability_masks(sample: Mapping[str, Any], target_valid: Tensor) -> dict[str, Tensor]:
    steps = target_valid.shape[0]
    physiology = sample.get("physiology_mask")
    if physiology is None:
        physiology = torch.ones((steps, 3), dtype=torch.bool)
    if not isinstance(physiology, Tensor) or physiology.shape != (steps, 3):
        raise ValueError(
            f"physiology_mask must have shape {(steps, 3)} in sample {sample.get('sample_id')!r}"
        )
    behavior = sample.get("modality_mask")
    if not isinstance(behavior, Tensor) or behavior.shape != (steps, 3):
        raise ValueError(
            f"modality_mask must have shape {(steps, 3)} in sample {sample.get('sample_id')!r}"
        )
    masks: dict[str, Tensor] = {}
    for index, modality in enumerate(PHYSIOLOGY_MODALITIES):
        masks[modality] = physiology[:, index].bool() & target_valid
    for index, modality in enumerate(BEHAVIOR_MODALITIES):
        masks[modality] = behavior[:, index].bool() & target_valid
    return masks


def select_minimal_cover(
    sample_coverages: Mapping[str, Sequence[str]], required_modalities: Sequence[str]
) -> tuple[str, ...]:
    """Return the deterministic exact minimum-cardinality modality cover."""

    required = tuple(sorted(set(required_modalities)))
    if not required:
        raise ValueError("At least one target-valid, enabled modality is required")
    bit_for = {name: 1 << index for index, name in enumerate(required)}
    full_mask = (1 << len(required)) - 1
    best: dict[int, tuple[str, ...]] = {0: ()}
    for sample_id in sorted(sample_coverages):
        if not sample_id:
            raise ValueError("Sample IDs must be non-empty")
        coverage_mask = 0
        for modality in sample_coverages[sample_id]:
            coverage_mask |= bit_for.get(modality, 0)
        if coverage_mask == 0:
            continue
        candidates = list(best.items())
        for previous_mask, previous_ids in candidates:
            combined_mask = previous_mask | coverage_mask
            combined_ids = (*previous_ids, sample_id)
            incumbent = best.get(combined_mask)
            if incumbent is None or (len(combined_ids), combined_ids) < (
                len(incumbent),
                incumbent,
            ):
                best[combined_mask] = combined_ids
    if full_mask not in best:
        covered = {
            modality
            for coverage in sample_coverages.values()
            for modality in coverage
            if modality in bit_for
        }
        missing = sorted(set(required) - covered)
        raise ValueError(f"No training-window cover exists for modalities: {missing}")
    return best[full_mask]


def _shared_audit_samples(
    configs: Sequence[Mapping[str, Any]], config_paths: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not configs:
        raise ValueError("At least one configuration is required")
    manifest_paths = [Path(str(config["data"]["train_manifest"])).resolve() for config in configs]
    if any(path != manifest_paths[0] for path in manifest_paths[1:]):
        details = [path.as_posix() for path in manifest_paths]
        raise ValueError(f"All compared configurations must use one train manifest: {details}")
    manifest_path = manifest_paths[0]
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha256 = _sha256_file(manifest_path)

    datasets = [ManifestDataset(str(path), require_targets=True) for path in manifest_paths]
    sample_id_lists = [
        tuple(str(record["sample_id"]) for record in dataset.records) for dataset in datasets
    ]
    if any(sample_ids != sample_id_lists[0] for sample_ids in sample_id_lists[1:]):
        raise ValueError("Compared configurations do not expose identical training sample IDs")
    sample_ids = sample_id_lists[0]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Training manifest contains duplicate sample IDs")

    enabled_by_config = {
        Path(path).as_posix(): list(_enabled_modalities(config))
        for path, config in zip(config_paths, configs, strict=True)
    }
    enabled_union = tuple(
        modality
        for modality in MODALITY_ORDER
        if any(modality in enabled for enabled in enabled_by_config.values())
    )
    aggregate_counts = dict.fromkeys(MODALITY_ORDER, 0)
    sample_coverages: dict[str, tuple[str, ...]] = {}
    sample_stats: dict[str, dict[str, Any]] = {}
    loaded_by_id: dict[str, dict[str, Any]] = {}
    for index, sample_id in enumerate(sample_ids):
        sample = datasets[0][index]
        target_valid = _target_valid_mask(sample)
        availability = _availability_masks(sample, target_valid)
        counts = {modality: int(mask.sum().item()) for modality, mask in availability.items()}
        for modality, count in counts.items():
            aggregate_counts[modality] += count
        coverage = tuple(modality for modality in enabled_union if counts.get(modality, 0) > 0)
        sample_coverages[sample_id] = coverage
        sample_stats[sample_id] = {
            "sample_id": sample_id,
            "target_valid_steps": int(target_valid.sum().item()),
            "target_valid_modality_steps": counts,
            "covered_enabled_modalities": list(coverage),
        }
        loaded_by_id[sample_id] = sample

    globally_available = tuple(
        modality for modality in MODALITY_ORDER if aggregate_counts[modality] > 0
    )
    required = tuple(modality for modality in enabled_union if modality in globally_available)
    selected_ids = select_minimal_cover(sample_coverages, required)
    selected_samples = [loaded_by_id[sample_id] for sample_id in selected_ids]
    selected_stats = [sample_stats[sample_id] for sample_id in selected_ids]

    tensor_files: dict[str, str] = {}
    record_by_id = {str(record["sample_id"]): record for record in datasets[0].records}
    for sample_id in selected_ids:
        tensor_path = Path(str(record_by_id[sample_id]["tensor_file"]))
        if not tensor_path.is_absolute():
            tensor_path = manifest_path.parent / tensor_path
        tensor_path = tensor_path.resolve()
        if not tensor_path.is_file():
            raise FileNotFoundError(tensor_path)
        tensor_files[sample_id] = _sha256_file(tensor_path)

    selection = {
        "selection_protocol": SELECTION_PROTOCOL,
        # Keep published audit artifacts machine-independent; the content hash
        # above, not an author workstation path, is the identity anchor.
        "manifest_path": Path(str(configs[0]["data"]["train_manifest"])).as_posix(),
        "manifest_sha256": manifest_sha256,
        "manifest_record_count": len(sample_ids),
        "manifest_sample_ids_sha256": _sha256_json(sample_ids),
        "enabled_modalities_by_config": enabled_by_config,
        "enabled_modalities_union": list(enabled_union),
        "globally_target_valid_modalities": list(globally_available),
        "required_cover_modalities": list(required),
        "enabled_but_globally_unavailable_modalities": sorted(
            set(enabled_union) - set(globally_available)
        ),
        # These are manifest-window occurrences.  Overlapping preparation
        # windows are intentionally not represented as unique timeline steps.
        "manifest_window_target_valid_modality_step_occurrences": aggregate_counts,
        "selected_sample_ids": list(selected_ids),
        "selected_sample_ids_sha256": _sha256_json(selected_ids),
        "selected_tensor_sha256": tensor_files,
        "selected_sample_count": len(selected_ids),
        "selected_samples": selected_stats,
    }
    return selected_samples, selection


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def parameter_record(
    config: Mapping[str, Any],
    config_path: str,
    samples: Sequence[dict[str, Any]],
    selection: Mapping[str, Any],
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(0)
    model = build_model(config).to(device)
    model.eval()
    criterion = EPTNetLoss(config).to(device)
    model.zero_grad(set_to_none=True)
    losses: list[float] = []
    for sample in samples:
        batch = _move_batch(collate_multimodal([sample]), device)
        loss = criterion(model(batch), batch)["total"] / len(samples)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError(f"Non-finite audit loss for {config_path} on {sample['sample_id']}")
        loss.backward()
        losses.append(float(loss.detach().cpu().item() * len(samples)))

    allocated = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    graph_participating = 0
    graph_parameter_tensors = 0
    nonzero_gradient_parameters = 0
    nonzero_gradient_parameter_tensors = 0
    nonzero_gradient_elements = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        graph_participating += parameter.numel()
        graph_parameter_tensors += 1
        finite = torch.isfinite(parameter.grad)
        if not bool(finite.all().item()):
            raise ValueError(f"Non-finite gradient in {config_path}:{name}")
        nonzero = parameter.grad != 0
        count = int(nonzero.sum().item())
        nonzero_gradient_elements += count
        if count > 0:
            nonzero_gradient_parameters += parameter.numel()
            nonzero_gradient_parameter_tensors += 1
    if graph_participating == 0 or nonzero_gradient_parameters == 0:
        raise ValueError(f"No effective gradient path was audited for {config_path}")

    selected_ids = list(selection["selected_sample_ids"])
    return {
        "config": _portable_repository_path(Path(config_path)),
        "experiment": config["experiment"]["name"],
        "model": config["model"]["name"],
        "allocated_parameters": allocated,
        "trainable_parameters": trainable,
        "graph_participating_parameters": graph_participating,
        "graph_participating_fraction": graph_participating / trainable,
        "graph_participating_parameter_tensors": graph_parameter_tensors,
        "nonzero_gradient_parameters": nonzero_gradient_parameters,
        "nonzero_gradient_parameter_fraction": nonzero_gradient_parameters / trainable,
        "nonzero_gradient_parameter_tensors": nonzero_gradient_parameter_tensors,
        "nonzero_gradient_elements": nonzero_gradient_elements,
        "nonzero_gradient_element_fraction": nonzero_gradient_elements / trainable,
        "zero_gradient_graph_parameters": graph_participating - nonzero_gradient_parameters,
        "mean_audit_loss": sum(losses) / len(losses),
        "audit_sample_ids": selected_ids,
        "audit_sample_ids_sha256": selection["selected_sample_ids_sha256"],
        "manifest_sha256": selection["manifest_sha256"],
        "counting_protocol": GRAPH_COUNTING_PROTOCOL,
        "nonzero_gradient_parameter_counting_protocol": (NONZERO_PARAMETER_COUNTING_PROTOCOL),
        "nonzero_gradient_element_counting_protocol": NONZERO_ELEMENT_COUNTING_PROTOCOL,
        # Backward-compatible aliases.  These are graph-participation counts,
        # not proof that every counted element received a non-zero gradient.
        "executed_parameters": graph_participating,
        "executed_fraction": graph_participating / trainable,
        "executed_parameters_is_alias_of": "graph_participating_parameters",
    }


def build_audit(config_paths: Sequence[str], device: torch.device) -> dict[str, Any]:
    configs = [load_config(path) for path in config_paths]
    audit_inputs = _build_audit_inputs(config_paths, configs)
    samples, selection = _shared_audit_samples(configs, config_paths)
    if selection["manifest_sha256"] != audit_inputs["train_manifest"]["sha256"]:
        raise ValueError("Training manifest changed while the parameter audit was starting")
    records = [
        parameter_record(config, path, samples, selection, device)
        for path, config in zip(config_paths, configs, strict=True)
    ]
    if _build_audit_inputs(config_paths, configs) != audit_inputs:
        raise ValueError("Audit inputs changed while the parameter audit was running")
    reference_graph = int(records[0]["graph_participating_parameters"])
    reference_nonzero_parameters = int(records[0]["nonzero_gradient_parameters"])
    reference_nonzero_elements = int(records[0]["nonzero_gradient_elements"])
    for record in records:
        graph = int(record["graph_participating_parameters"])
        nonzero_parameters = int(record["nonzero_gradient_parameters"])
        nonzero_elements = int(record["nonzero_gradient_elements"])
        record["graph_participating_ratio_to_eptnet"] = graph / reference_graph
        record["nonzero_gradient_parameter_ratio_to_eptnet"] = (
            nonzero_parameters / reference_nonzero_parameters
        )
        record["nonzero_gradient_element_ratio_to_eptnet"] = (
            nonzero_elements / reference_nonzero_elements
        )
        record["executed_parameter_ratio_to_eptnet"] = graph / reference_graph
    return {
        "reference_experiment": records[0]["experiment"],
        "device": str(device),
        "deterministic_model_seed": 0,
        "audit_inputs": audit_inputs,
        "selection": selection,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit shared-sample graph and non-zero-gradient parameter fairness"
    )
    parser.add_argument("configs", nargs="*", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/parameter_fairness.json")
    parser.add_argument(
        "--check-reference",
        help=(
            "read-only validation of an existing audit against the exact current "
            "config, implementation-source, and train-manifest fingerprints"
        ),
    )
    args = parser.parse_args()

    if args.check_reference:
        result = check_audit_reference(args.configs, Path(args.check_reference))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return

    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite existing parameter-fairness audit: {output}"
        )
    device = resolve_device(args.device)
    payload = build_audit(args.configs, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
