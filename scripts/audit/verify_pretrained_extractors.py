"""Audit entry point for frozen pretrained behavior extractors."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from eptnet.data.behavior_features import (
    MACBERT_MODEL_ID,
    MACBERT_REVISION,
    WAVLM_MODEL_ID,
    WAVLM_REVISION,
    MacBERTTextEncoder,
    WavLMFeatureEncoder,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen official behavior extractors")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model-cache", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/pretrained_extractor_verification.json")
    args = parser.parse_args()

    text_encoder = MacBERTTextEncoder(
        model_id=MACBERT_MODEL_ID,
        revision=MACBERT_REVISION,
        device=args.device,
        batch_size=2,
        cache_dir=args.model_cache,
        local_files_only=True,
    )
    text = text_encoder.encode(["这是诚实回答", "这段陈述需要进一步核验"])

    audio_encoder = WavLMFeatureEncoder(
        model_id=WAVLM_MODEL_ID,
        revision=WAVLM_REVISION,
        device=args.device,
        batch_size=2,
        cache_dir=args.model_cache,
        local_files_only=True,
    )
    audio, audio_mask = audio_encoder.encode_file(Path(args.audio), num_steps=2, step_seconds=1.0)

    if text.shape != (2, 768) or not np.isfinite(text).all():
        raise ValueError(f"MacBERT verification failed with shape {text.shape}")
    if audio.shape != (2, 768) or not np.isfinite(audio).all() or not audio_mask.all():
        raise ValueError(
            f"WavLM verification failed with shape {audio.shape} and mask {audio_mask.tolist()}"
        )
    report = {
        "status": "passed",
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
        "macbert": {
            "model_id": MACBERT_MODEL_ID,
            "revision": MACBERT_REVISION,
            "output_shape": list(text.shape),
            "finite": True,
            "pooling": "last_hidden_state_cls",
        },
        "wavlm": {
            "model_id": WAVLM_MODEL_ID,
            "revision": WAVLM_REVISION,
            "authorized_local_audio_used": True,
            "steps_checked": 2,
            "output_shape": list(audio.shape),
            "availability_mask": audio_mask.tolist(),
            "finite": True,
            "pooling": "per-step masked mean of last_hidden_state",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
