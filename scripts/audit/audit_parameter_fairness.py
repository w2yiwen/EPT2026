"""Audit entry point for executed-model parameter fairness."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
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
        "config": Path(config_path).as_posix(),
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
    samples, selection = _shared_audit_samples(configs, config_paths)
    records = [
        parameter_record(config, path, samples, selection, device)
        for path, config in zip(config_paths, configs, strict=True)
    ]
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
    args = parser.parse_args()

    device = resolve_device(args.device)
    payload = build_audit(args.configs, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
