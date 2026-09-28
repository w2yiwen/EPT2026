#!/usr/bin/env python3
"""Run bounded local checks for OpenFace, WavLM, MacBERT, and shared Causal TCN."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eptnet.models.behavior import (
    CausalTCNDecoder,
    MacBERTBaseEncoder,
    OpenFaceFeatureEncoder,
    WavLMBasePlusEncoder,
)


def _decode_check(features: np.ndarray, mask: np.ndarray, device: torch.device) -> dict[str, Any]:
    decoder = CausalTCNDecoder(features.shape[1], dropout=0.0).to(device).eval()
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    sequence_mask = torch.from_numpy(mask).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = decoder(tensor, sequence_mask)
    return {
        "input_shape": list(tensor.shape),
        "valid_steps": int(sequence_mask.sum().item()),
        "parameter_count": sum(parameter.numel() for parameter in decoder.parameters()),
        "class_logits_shape": list(outputs["class_logits"].shape),
        "finite": all(
            bool(torch.isfinite(value).all().item())
            for value in outputs.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--openface-video", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/behavior_encoder_baseline_smoke.json")
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "environment": {
            "python": __import__("sys").version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
        },
        "full_experiment_started": False,
        "modalities": {},
    }

    face_encoder = OpenFaceFeatureEncoder()
    if args.openface_video is None:
        payload["modalities"]["face"] = {
            "status": "not_run",
            "reason": "--openface-video was not provided",
            "runtime_available": face_encoder.executable is not None,
        }
    else:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="eptnet-openface-smoke-") as temporary:
            csv_path = face_encoder.extract_video(args.openface_video, temporary)
            encoded_face = face_encoder.encode_csv(csv_path)
        payload["modalities"]["face"] = {
            "status": "passed",
            "encoder": "OpenFace 2.2.0",
            "feature_dim": face_encoder.output_dim,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "decoder": _decode_check(encoded_face.features, encoded_face.mask, device),
        }

    sample_rate = WavLMBasePlusEncoder.sample_rate
    seconds = 2
    time_axis = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
    waveform = (0.05 * np.sin(2 * np.pi * 220.0 * time_axis)).astype(np.float32)
    started = time.perf_counter()
    audio_encoder = WavLMBasePlusEncoder(device=str(device), batch_size=2)
    encoded_audio = audio_encoder.encode_waveform(waveform, num_steps=seconds)
    payload["modalities"]["audio"] = {
        "status": "passed",
        "encoder": "microsoft/wavlm-base-plus",
        "feature_dim": audio_encoder.output_dim,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "decoder": _decode_check(encoded_audio.features, encoded_audio.mask, device),
    }
    del audio_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    started = time.perf_counter()
    text_encoder = MacBERTBaseEncoder(device=str(device), batch_size=2)
    encoded_text = text_encoder.encode(
        ["受试者正在回答当前问题。", "这是一段可复现的文本编码测试。"]
    )
    payload["modalities"]["text"] = {
        "status": "passed",
        "encoder": "hfl/chinese-macbert-base",
        "feature_dim": text_encoder.output_dim,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "decoder": _decode_check(encoded_text.features, encoded_text.mask, device),
    }
    del text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    payload["all_executed_modalities_passed"] = all(
        item["status"] == "passed"
        for item in payload["modalities"].values()
        if item["status"] != "not_run"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
