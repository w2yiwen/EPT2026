from __future__ import annotations

import math
import os
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..common import EncodedSequence

WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"
WAVLM_MODEL_REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
WAVLM_SOURCE_COMMIT = "31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd"


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_audio_mono(path: Path, sample_rate: int) -> np.ndarray:
    try:
        import soundfile as sf

        waveform, source_rate = sf.read(path, always_2d=True, dtype="float32")
        waveform = waveform.mean(axis=1)
    except Exception as soundfile_error:
        try:
            import librosa

            waveform, source_rate = librosa.load(path, sr=None, mono=True)
        except Exception as librosa_error:
            raise RuntimeError(
                f"Could not decode {path}; soundfile={soundfile_error!s}"
            ) from librosa_error
    waveform = np.asarray(waveform, dtype=np.float32)
    if int(source_rate) != sample_rate:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(source_rate), sample_rate)
        waveform = resample_poly(
            waveform, sample_rate // divisor, int(source_rate) // divisor
        ).astype(np.float32)
    return np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)


class WavLMBasePlusEncoder:
    """Frozen official WavLM Base+ with causal fixed-window pooling."""

    output_dim = 768
    sample_rate = 16_000

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        batch_size: int = 8,
        allow_remote: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        import torch
        from transformers import AutoFeatureExtractor, WavLMModel

        self._torch = torch
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        local_assets = Path(__file__).resolve().parent / "wavlm_base_plus" / "assets"
        requested = (
            Path(model_path).expanduser().resolve() if model_path is not None else local_assets
        )
        if (requested / "config.json").is_file():
            source = str(requested)
            load_options = {"local_files_only": True}
        elif allow_remote:
            source = WAVLM_MODEL_ID
            load_options = {"revision": WAVLM_MODEL_REVISION}
        else:
            raise FileNotFoundError(
                f"WavLM assets are missing from {requested}; run the model asset installer"
            )
        self.processor = AutoFeatureExtractor.from_pretrained(source, **load_options)
        self.model = WavLMModel.from_pretrained(source, **load_options).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def encode_waveform(
        self,
        waveform: np.ndarray,
        *,
        num_steps: int | None = None,
        step_seconds: float = 1.0,
    ) -> EncodedSequence:
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        signal = np.nan_to_num(
            np.asarray(waveform, dtype=np.float32).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        samples_per_step = max(1, int(round(step_seconds * self.sample_rate)))
        inferred_steps = int(math.ceil(len(signal) / samples_per_step)) if len(signal) else 0
        steps = inferred_steps if num_steps is None else int(num_steps)
        if steps < 0:
            raise ValueError("num_steps must be non-negative")
        output = np.zeros((steps, self.output_dim), dtype=np.float32)
        mask = np.zeros(steps, dtype=bool)
        chunks: list[np.ndarray] = []
        indices: list[int] = []
        for step in range(steps):
            left = step * samples_per_step
            right = min(left + samples_per_step, len(signal))
            if right - left < 400:
                continue
            chunks.append(signal[left:right])
            indices.append(step)
        for start in range(0, len(chunks), self.batch_size):
            batch_chunks: Sequence[np.ndarray] = chunks[start : start + self.batch_size]
            batch_indices = indices[start : start + self.batch_size]
            encoded_inputs = self.processor(
                list(batch_chunks),
                sampling_rate=self.sample_rate,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            encoded_inputs = {name: value.to(self.device) for name, value in encoded_inputs.items()}
            with self._torch.inference_mode():
                # transformers 4.56 passes a boolean padding mask together with
                # WavLM's floating-point relative-position bias to PyTorch 2.7.
                # The combination is valid today but emits an upstream deprecation
                # warning. Keep the suppression exact and local to this call.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=(
                            "Support for mismatched key_padding_mask and attn_mask is deprecated.*"
                        ),
                        category=UserWarning,
                        module="torch.nn.functional",
                    )
                    hidden = self.model(**encoded_inputs).last_hidden_state
            attention_mask = encoded_inputs.get("attention_mask")
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                frame_mask = self.model._get_feature_vector_attention_mask(
                    hidden.shape[1], attention_mask
                )
                weights = frame_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            output[np.asarray(batch_indices)] = pooled.float().cpu().numpy()
            mask[np.asarray(batch_indices)] = True
        output[~mask] = 0.0
        return EncodedSequence(output, mask).validate(output_dim=self.output_dim)

    def encode_file(
        self,
        path: str | Path,
        *,
        num_steps: int | None = None,
        step_seconds: float = 1.0,
    ) -> EncodedSequence:
        waveform = _load_audio_mono(Path(path).expanduser().resolve(), self.sample_rate)
        return self.encode_waveform(waveform, num_steps=num_steps, step_seconds=step_seconds)
