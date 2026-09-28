from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from eptnet.models.behavior.audio.wavlm import WavLMBasePlusEncoder
from eptnet.models.behavior.face.marlin import (
    MARLIN_OUTPUT_DIM,
    MARLIN_PREPROCESSING_VERSION,
    MarlinFeatureEncoder,
)
from eptnet.models.behavior.face.openface import (
    OPENFACE_FEATURE_NAMES,
    OpenFaceFeatureEncoder,
)
from eptnet.models.behavior.face.openface import (
    OPENFACE_FRAME_COLUMNS as OPENFACE_FRAME_COLUMNS,
)
from eptnet.models.behavior.text.macbert import MacBERTBaseEncoder

MACBERT_MODEL_ID = "hfl/chinese-macbert-base"
MACBERT_REVISION = "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1"
WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"
WAVLM_REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"

class TextEncoder(Protocol):
    output_dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class AudioEncoder(Protocol):
    output_dim: int

    def encode_file(
        self, path: Path, *, num_steps: int, step_seconds: float
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class BehaviorFeatureConfig:
    """Frozen upstream behavior-feature contract used during offline preparation.

    Heavy pretrained models are optional and loaded only when their backend is
    selected. Pinning model revisions here makes a prepared dataset independently
    auditable even when a model hub later updates its default branch.
    """

    video_backend: Literal["legacy", "marlin", "openface"] = "legacy"
    audio_backend: Literal["none", "wavlm"] = "none"
    text_backend: Literal["hash", "macbert"] = "hash"
    text_model_id: str = MACBERT_MODEL_ID
    text_revision: str = MACBERT_REVISION
    audio_model_id: str = WAVLM_MODEL_ID
    audio_revision: str = WAVLM_REVISION
    device: str = "auto"
    batch_size: int = 16
    cache_dir: str | None = None
    marlin_checkpoint: str | None = None
    marlin_crop_face: bool = True
    marlin_cache_dir: str | None = None
    openface_cache_dir: str | None = None
    local_files_only: bool = False
    openface_min_confidence: float = 0.8

    @property
    def video_dim(self) -> int:
        if self.video_backend == "legacy":
            return 3
        if self.video_backend == "marlin":
            return MARLIN_OUTPUT_DIM
        return len(OPENFACE_FEATURE_NAMES)

    @property
    def audio_dim(self) -> int:
        return 50 if self.audio_backend == "none" else 768

    @property
    def text_dim(self) -> int:
        return 768

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("behavior batch_size must be positive")
        if not 0.0 <= self.openface_min_confidence <= 1.0:
            raise ValueError("openface_min_confidence must lie in [0, 1]")
        for name in (
            "text_model_id",
            "text_revision",
            "audio_model_id",
            "audio_revision",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")

    def provenance(self) -> dict[str, object]:
        payload = asdict(self)
        payload["feature_dimensions"] = {
            "video": self.video_dim,
            "audio": self.audio_dim,
            "text": self.text_dim,
        }
        if self.video_backend == "marlin":
            payload["video_feature_order"] = [
                f"marlin_{index:03d}" for index in range(MARLIN_OUTPUT_DIM)
            ]
            payload["video_pooling"] = "official_marlin_clip_mean_pool"
            payload["video_preprocessing"] = MARLIN_PREPROCESSING_VERSION
        elif self.video_backend == "openface":
            payload["openface_feature_order"] = list(OPENFACE_FEATURE_NAMES)
        payload["text_pooling"] = "non_special_token_masked_mean"
        payload["audio_pooling"] = "per-step masked mean of last_hidden_state"
        return payload


def _resolved_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolved_cache_dir(cache_dir: str | None) -> str | None:
    """Accept either a Transformers cache directory or a Hugging Face home."""
    if cache_dir is None:
        return None
    root = Path(cache_dir).expanduser()
    hub = root / "hub"
    return str(hub if hub.is_dir() else root)


class MacBERTTextEncoder:
    """Official HFL MacBERT-base encoder with deterministic CLS pooling."""

    output_dim = 768

    def __init__(
        self,
        *,
        model_id: str = MACBERT_MODEL_ID,
        revision: str = MACBERT_REVISION,
        device: str = "auto",
        batch_size: int = 16,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        max_length: int = 128,
    ) -> None:
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        import torch
        from transformers import BertModel, BertTokenizer

        self._torch = torch
        self.device = _resolved_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        load_options = {
            "revision": revision,
            "cache_dir": _resolved_cache_dir(cache_dir),
            "local_files_only": local_files_only,
        }
        self.tokenizer = BertTokenizer.from_pretrained(model_id, **load_options)
        self.model = BertModel.from_pretrained(model_id, **load_options).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        output = np.zeros((len(texts), self.output_dim), dtype=np.float32)
        non_empty = [index for index, text in enumerate(texts) if text.strip()]
        for start in range(0, len(non_empty), self.batch_size):
            indices = non_empty[start : start + self.batch_size]
            batch_text = [texts[index] for index in indices]
            tokens = self.tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {name: value.to(self.device) for name, value in tokens.items()}
            with self._torch.inference_mode():
                encoded = self.model(**tokens).last_hidden_state[:, 0]
            output[np.asarray(indices)] = encoded.float().cpu().numpy()
        return output


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
                f"Could not decode audio file {path}; install soundfile with MP3 support "
                f"or librosa/ffmpeg (soundfile: {soundfile_error!s})"
            ) from librosa_error
    waveform = np.asarray(waveform, dtype=np.float32)
    if source_rate != sample_rate:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(source_rate), sample_rate)
        waveform = resample_poly(
            waveform,
            sample_rate // divisor,
            int(source_rate) // divisor,
        ).astype(np.float32)
    return np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)


class WavLMFeatureEncoder:
    """Official Microsoft WavLM Base+ as a frozen, per-step audio encoder."""

    output_dim = 768
    sample_rate = 16_000

    def __init__(
        self,
        *,
        model_id: str = WAVLM_MODEL_ID,
        revision: str = WAVLM_REVISION,
        device: str = "auto",
        batch_size: int = 8,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        import torch
        from transformers import AutoFeatureExtractor, WavLMModel

        self._torch = torch
        self.device = _resolved_device(device)
        self.batch_size = batch_size
        load_options = {
            "revision": revision,
            "cache_dir": _resolved_cache_dir(cache_dir),
            "local_files_only": local_files_only,
        }
        self.processor = AutoFeatureExtractor.from_pretrained(model_id, **load_options)
        self.model = WavLMModel.from_pretrained(model_id, **load_options).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def encode_file(
        self, path: Path, *, num_steps: int, step_seconds: float
    ) -> tuple[np.ndarray, np.ndarray]:
        waveform = _load_audio_mono(path, self.sample_rate)
        samples_per_step = max(1, int(round(step_seconds * self.sample_rate)))
        output = np.zeros((num_steps, self.output_dim), dtype=np.float32)
        mask = np.zeros(num_steps, dtype=bool)
        chunks: list[np.ndarray] = []
        indices: list[int] = []
        for step in range(num_steps):
            left = step * samples_per_step
            right = min(left + samples_per_step, len(waveform))
            if right <= left:
                continue
            chunks.append(waveform[left:right])
            indices.append(step)
        for start in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[start : start + self.batch_size]
            batch_indices = indices[start : start + self.batch_size]
            encoded_inputs = self.processor(
                batch_chunks,
                sampling_rate=self.sample_rate,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            encoded_inputs = {name: value.to(self.device) for name, value in encoded_inputs.items()}
            with self._torch.inference_mode():
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
        return output, mask


def extract_openface_features(
    path: Path | None,
    *,
    num_steps: int,
    min_confidence: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool official OpenFace frame CSV output into causal one-step features.

    The first half contains per-step means and the second half per-step standard
    deviations. Rows that OpenFace marks unsuccessful or low-confidence are
    excluded. Timestamps are duration-normalized to match the existing acquisition
    policy, whose device clocks are incomplete.
    """

    if path is None:
        return (
            np.zeros((num_steps, len(OPENFACE_FEATURE_NAMES)), dtype=np.float32),
            np.zeros(num_steps, dtype=bool),
        )
    encoded = OpenFaceFeatureEncoder(min_confidence=min_confidence).encode_csv(
        path, num_steps=num_steps
    )
    return encoded.features, encoded.mask


class BehaviorFeaturePipeline:
    """Lazy adapter around the selected behavior backbones."""

    def __init__(self, config: BehaviorFeatureConfig) -> None:
        config.validate()
        self.config = config
        self._text_encoder: TextEncoder | None = None
        self._audio_encoder: AudioEncoder | None = None
        self._face_encoder: MarlinFeatureEncoder | None = None
        self.last_video_alignment_method: str | None = None

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        if self.config.text_backend == "hash":
            raise RuntimeError("hash text features are implemented by the legacy adapter")
        if self._text_encoder is None:
            self._text_encoder = MacBERTBaseEncoder(
                device=self.config.device,
                batch_size=self.config.batch_size,
            )
        encoded = self._text_encoder.encode(texts)
        return encoded.features if hasattr(encoded, "features") else encoded

    def encode_audio(
        self, path: Path | None, *, num_steps: int, step_seconds: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.config.audio_backend == "none" or path is None:
            return (
                np.zeros((num_steps, self.config.audio_dim), dtype=np.float32),
                np.zeros(num_steps, dtype=bool),
            )
        if self._audio_encoder is None:
            self._audio_encoder = WavLMBasePlusEncoder(
                device=self.config.device,
                batch_size=max(1, self.config.batch_size // 2),
            )
        encoded = self._audio_encoder.encode_file(
            path, num_steps=num_steps, step_seconds=step_seconds
        )
        if hasattr(encoded, "features"):
            return encoded.features, encoded.mask
        return encoded

    def encode_video(
        self,
        path: Path | None,
        *,
        num_steps: int,
        step_seconds: float | None = None,
        time_offset_seconds: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.config.video_backend == "marlin":
            if path is None:
                return (
                    np.zeros((num_steps, MARLIN_OUTPUT_DIM), dtype=np.float32),
                    np.zeros(num_steps, dtype=bool),
                )
            if path.suffix.lower() == ".csv":
                raise ValueError("MARLIN requires a raw video file, not a facial-feature CSV")
            cache_root = (
                Path(self.config.marlin_cache_dir).expanduser().resolve()
                if self.config.marlin_cache_dir
                else None
            )
            cached_features: Path | None = None
            remapped_features: Path | None = None
            if cache_root is not None:
                cache_root.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                cache_key = (
                    f"{path.stem}.{digest.hexdigest()[:16]}.steps{num_steps}."
                    f"step{step_seconds}.offset{time_offset_seconds:.6f}."
                    f"crop{int(self.config.marlin_crop_face)}."
                    f"{MARLIN_PREPROCESSING_VERSION}"
                )
                cached_features = cache_root / f"{cache_key}.npz"
                remapped_key = cache_key.rsplit(".", 1)[0] + ".legacy_embedding_time_remap_v1"
                remapped_features = cache_root / f"{remapped_key}.npz"
                if not cached_features.is_file() and remapped_features.is_file():
                    cached_features = remapped_features
            if cached_features is not None and cached_features.is_file():
                with np.load(cached_features, allow_pickle=False) as cached:
                    features = np.asarray(cached["features"], dtype=np.float32)
                    mask = np.asarray(cached["mask"], dtype=bool)
                if features.shape != (num_steps, MARLIN_OUTPUT_DIM) or mask.shape != (
                    num_steps,
                ):
                    raise RuntimeError(f"Invalid MARLIN cache shape: {cached_features}")
                print(f"MARLIN cache hit: {cached_features}", flush=True)
                self.last_video_alignment_method = (
                    "legacy_embedding_time_remap_v1"
                    if "legacy_embedding_time_remap_v1" in cached_features.name
                    else "direct_real_time_frame_sampling_v2"
                )
                return features, mask
            if self._face_encoder is None:
                self._face_encoder = MarlinFeatureEncoder(
                    checkpoint=self.config.marlin_checkpoint,
                    device=self.config.device,
                    batch_size=max(1, min(16, self.config.batch_size)),
                    crop_face=self.config.marlin_crop_face,
                    asset_dir=self.config.cache_dir,
                )
            encoded = (
                self._face_encoder.encode_video(path, num_steps=num_steps)
                if step_seconds is None
                else self._face_encoder.encode_video(
                    path,
                    num_steps=num_steps,
                    step_seconds=step_seconds,
                    time_offset_seconds=time_offset_seconds,
                )
            )
            self.last_video_alignment_method = "direct_real_time_frame_sampling_v2"
            if cached_features is not None:
                temporary = cached_features.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temporary,
                    features=encoded.features,
                    mask=encoded.mask,
                )
                temporary.replace(cached_features)
                print(f"MARLIN cache written: {cached_features}", flush=True)
            return encoded.features, encoded.mask
        if self.config.video_backend != "openface":
            raise RuntimeError("legacy facial features are implemented by the legacy adapter")
        if path is None:
            return extract_openface_features(
                None,
                num_steps=num_steps,
                min_confidence=self.config.openface_min_confidence,
            )
        if path.suffix.lower() == ".csv":
            return extract_openface_features(
                path,
                num_steps=num_steps,
                min_confidence=self.config.openface_min_confidence,
            )
        encoder = OpenFaceFeatureEncoder(min_confidence=self.config.openface_min_confidence)
        cache_root = (
            Path(self.config.openface_cache_dir).expanduser().resolve()
            if self.config.openface_cache_dir
            else None
        )
        cached_csv: Path | None = None
        if cache_root is not None:
            cache_root.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            cached_csv = cache_root / f"{path.stem}.{digest.hexdigest()[:16]}.csv"
        cache_hit = cached_csv is not None and cached_csv.is_file()
        if cache_hit:
            csv_path = cached_csv
        else:
            temporary_parent = str(cache_root) if cache_root is not None else None
            with tempfile.TemporaryDirectory(
                prefix="eptnet-openface-", dir=temporary_parent
            ) as temporary:
                generated = encoder.extract_video(path, temporary)
                if cached_csv is None:
                    encoded = encoder.encode_csv(generated, num_steps=num_steps)
                    return encoded.features, encoded.mask
                generated.replace(cached_csv)
            csv_path = cached_csv
        print(
            f"OpenFace cache {'hit' if cache_hit else 'written'}: {csv_path}",
            flush=True,
        )
        encoded = encoder.encode_csv(csv_path, num_steps=num_steps)
        return encoded.features, encoded.mask
