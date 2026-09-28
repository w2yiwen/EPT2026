#!/usr/bin/env python3
"""Run bounded EEGNet/NeuroKit2 encoder checks without training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from eptnet.models.physiology import (  # noqa: E402
    EEGNetFeatureEncoder,
    EEGNetPreprocessor,
    NeuroKitPPGEncoder,
)
from eptnet.provenance import portable_path  # noqa: E402


def _synthetic_ppg(seconds: int, sampling_rate: int) -> np.ndarray:
    time_axis = np.arange(seconds * sampling_rate, dtype=np.float64) / sampling_rate
    carrier = np.sin(2 * np.pi * 1.15 * time_axis - np.pi / 2)
    morphology = 0.22 * np.sin(2 * np.pi * 2.3 * time_axis - np.pi / 4)
    return np.stack(
        (
            carrier + morphology + 0.03 * np.sin(2 * np.pi * 0.2 * time_axis),
            0.9 * carrier + 0.18 * morphology + 0.02 * np.cos(2 * np.pi * 0.3 * time_axis),
        )
    )


def _read_openbci_window(path: Path, seconds: float) -> tuple[np.ndarray, float]:
    sampling_rate = 500.0
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("%"):
                match = re.search(r"Sample Rate\s*=\s*([0-9.]+)", line)
                if match:
                    sampling_rate = float(match.group(1))
                continue
            fields = [value.strip() for value in line.split(",")]
            if len(fields) < 9:
                continue
            try:
                rows.append([float(value) for value in fields[1:9]])
            except ValueError:
                continue
            if len(rows) >= int(np.ceil(sampling_rate * seconds)):
                break
    if len(rows) < int(np.ceil(sampling_rate * seconds)):
        raise ValueError(f"{path} does not contain a complete {seconds:g}-second EEG window")
    return np.asarray(rows, dtype=np.float64).T, sampling_rate


def _read_scalar_prefix(path: Path, samples: int) -> np.ndarray:
    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = float(line.strip())
            except ValueError:
                continue
            if np.isfinite(value):
                values.append(value)
            if len(values) >= samples:
                break
    if len(values) < samples:
        raise ValueError(f"{path} has {len(values)} values; expected {samples}")
    return np.asarray(values, dtype=np.float64)


def _finite(value: torch.Tensor | np.ndarray) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    return bool(np.isfinite(value).all())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eeg-file", type=Path)
    parser.add_argument("--ppg-channel-1", type=Path)
    parser.add_argument("--ppg-channel-2", type=Path)
    parser.add_argument("--ppg-sampling-rate", type=float, default=385.0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/physiology_encoder_baseline_smoke.json")
    )
    args = parser.parse_args()
    if (args.ppg_channel_1 is None) != (args.ppg_channel_2 is None):
        parser.error("both PPG channels must be supplied together")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    payload: dict[str, Any] = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_experiment_started": False,
        "training_started": False,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "environment": {
            "python": __import__("sys").version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "neurokit2": importlib.metadata.version("neurokit2"),
        },
        "encoders": {},
    }

    torch.manual_seed(7)
    eeg_encoder = EEGNetFeatureEncoder(drop_prob=0.0).to(device).eval()
    synthetic_eeg = torch.randn(2, 3, 8, 200, device=device)
    started = time.perf_counter()
    with torch.inference_mode():
        eeg_features = eeg_encoder(synthetic_eeg)
        eeg_logits = eeg_encoder.forward_logits(synthetic_eeg)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    payload["encoders"]["eegnet"] = {
        "status": "passed",
        "source": "Braindecode EEGNet-v4 core",
        "weights": "random initialization; train from scratch baseline",
        "input_shape": list(synthetic_eeg.shape),
        "feature_shape": list(eeg_features.shape),
        "logits_shape": list(eeg_logits.shape),
        "feature_dim": eeg_encoder.output_dim,
        "parameter_count": eeg_encoder.parameter_count,
        "finite": _finite(eeg_features) and _finite(eeg_logits),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }

    if args.eeg_file is not None:
        raw_eeg, source_rate = _read_openbci_window(args.eeg_file.resolve(), 1.0)
        processed_eeg = EEGNetPreprocessor()(raw_eeg, source_rate_hz=source_rate)
        tensor = torch.from_numpy(processed_eeg).unsqueeze(0).to(device)
        with torch.inference_mode():
            real_features = eeg_encoder(tensor)
            real_logits = eeg_encoder.forward_logits(tensor)
        payload["encoders"]["eegnet"]["real_input"] = {
            "source": portable_path(args.eeg_file, root=PROJECT_ROOT),
            "native_sampling_rate_hz": source_rate,
            "processed_shape": list(processed_eeg.shape),
            "processed_dtype": str(processed_eeg.dtype),
            "feature_shape": list(real_features.shape),
            "logits_shape": list(real_logits.shape),
            "finite": _finite(real_features) and _finite(real_logits),
        }

    ppg_encoder = NeuroKitPPGEncoder(backend="neurokit2")
    synthetic_ppg = _synthetic_ppg(14, 385)
    started = time.perf_counter()
    encoded_ppg = ppg_encoder.encode(synthetic_ppg, sampling_rate_hz=385)
    payload["encoders"]["neurokit2_ppg"] = {
        "status": "passed",
        "source": "NeuroKit2 0.2.11 Elgendi cleaner and peak detector",
        "input_shape": list(synthetic_ppg.shape),
        "feature_shape": list(encoded_ppg.features.shape),
        "channel_mask_shape": list(encoded_ppg.channel_mask.shape),
        "feature_dim": ppg_encoder.output_dim,
        "valid_steps": int(encoded_ppg.mask.sum()),
        "feature_names_per_channel": list(ppg_encoder.feature_names),
        "finite": _finite(encoded_ppg.features),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }

    if args.ppg_channel_1 is not None and args.ppg_channel_2 is not None:
        seconds = 14
        samples = int(round(seconds * args.ppg_sampling_rate))
        real_ppg = np.stack(
            (
                _read_scalar_prefix(args.ppg_channel_1.resolve(), samples),
                _read_scalar_prefix(args.ppg_channel_2.resolve(), samples),
            )
        )
        real_encoded = ppg_encoder.encode(
            real_ppg, sampling_rate_hz=args.ppg_sampling_rate
        )
        payload["encoders"]["neurokit2_ppg"]["real_input"] = {
            "sources": [
                portable_path(args.ppg_channel_1, root=PROJECT_ROOT),
                portable_path(args.ppg_channel_2, root=PROJECT_ROOT),
            ],
            "sampling_rate_hz": args.ppg_sampling_rate,
            "input_shape": list(real_ppg.shape),
            "feature_shape": list(real_encoded.features.shape),
            "valid_steps": int(real_encoded.mask.sum()),
            "valid_channel_steps": real_encoded.channel_mask.sum(axis=0).astype(int).tolist(),
            "finite": _finite(real_encoded.features),
        }

    payload["all_checks_passed"] = all(
        item["status"] == "passed" and item["finite"] for item in payload["encoders"].values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
