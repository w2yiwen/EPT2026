from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from ..common import EncodedSequence

MARLIN_MODEL_NAME = "marlin_vit_small_ytf"
MARLIN_OUTPUT_DIM = 384
MARLIN_SOURCE_COMMIT = "c9d5698f98799828b5bcd0528e408a06e3f58502"
MARLIN_VERSION = "0.3.4"
MARLIN_PREPROCESSING_VERSION = "per_sampled_frame_face_detection_real_time_v2"
FACEXZOO_RETINAFACE_SHA256 = (
    "c3dab9a03d12296818ba25d7d1a60ff4fb98bf42523ce5f53552ed246013b52b"
)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class MarlinFeatureEncoder:
    """Frozen official MARLIN-small facial-video feature extractor.

    The adapter only owns video decoding, temporal binning, batching, and the
    project's ``EncodedSequence`` contract. Model construction and feature
    extraction are delegated to the authors' ``marlin-pytorch`` package.
    """

    output_dim = MARLIN_OUTPUT_DIM
    clip_frames = 16
    image_size = 224

    def __init__(
        self,
        *,
        model_name: str = MARLIN_MODEL_NAME,
        checkpoint: str | Path | None = None,
        device: str = "auto",
        batch_size: int = 2,
        crop_face: bool = True,
        use_amp: bool = True,
        asset_dir: str | Path | None = None,
        model: object | None = None,
    ) -> None:
        if model_name != MARLIN_MODEL_NAME:
            raise ValueError(f"Unsupported MARLIN model: {model_name}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.checkpoint = Path(checkpoint).expanduser().resolve() if checkpoint else None
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.crop_face = bool(crop_face)
        self.use_amp = bool(use_amp)
        self.asset_dir = (
            Path(asset_dir).expanduser().resolve()
            if asset_dir
            else Path(__file__).resolve().parent / "marlin" / "assets"
        )
        self._model = model

    @property
    def model(self):
        if self._model is None:
            try:
                from marlin_pytorch import Marlin
            except ImportError as error:
                raise RuntimeError(
                    "MARLIN requires marlin-pytorch==0.3.4; install requirements-behavior.txt"
                ) from error
            local_checkpoint = self.checkpoint
            if local_checkpoint is None:
                candidate = self.asset_dir / f"{self.model_name}.encoder.pt"
                local_checkpoint = candidate if candidate.is_file() else None
            self._model = (
                Marlin.from_file(self.model_name, str(local_checkpoint))
                if local_checkpoint is not None
                else Marlin.from_online(self.model_name)
            )
        self._model.to(self.device)
        self._model.eval()
        self._model.requires_grad_(False)
        return self._model

    def _face_detector(self):
        try:
            from marlin_pytorch.face_detector import FaceXZooFaceDetector
        except ImportError as error:
            raise RuntimeError("The official MARLIN face detector is unavailable") from error
        if not FaceXZooFaceDetector.inited:
            self.asset_dir.mkdir(parents=True, exist_ok=True)
            sdk = FaceXZooFaceDetector.install(str(self.asset_dir / "FaceX-Zoo"))
            if sdk not in sys.path:
                sys.path.append(sdk)
            detector_checkpoint = (
                Path(sdk)
                / "models"
                / "face_detection"
                / "face_detection_1.0"
                / "face_detection_retina.pkl"
            )
            actual_sha256 = hashlib.sha256(detector_checkpoint.read_bytes()).hexdigest()
            if actual_sha256 != FACEXZOO_RETINAFACE_SHA256:
                raise RuntimeError(
                    "Refusing to deserialize an unverified FaceX-Zoo checkpoint: "
                    f"{detector_checkpoint}"
                )

            # PyTorch 2.6 made ``weights_only=True`` the default, but this verified
            # upstream checkpoint contains the RetinaFace model object. Scope the
            # legacy loader to FaceX-Zoo initialization and immediately restore it.
            import torch

            original_torch_load = torch.load

            def trusted_facexzoo_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return original_torch_load(*args, **kwargs)

            torch.load = trusted_facexzoo_load
            try:
                FaceXZooFaceDetector.init(face_sdk_path=sdk, device=self.device)
            finally:
                torch.load = original_torch_load
        return FaceXZooFaceDetector

    def _prepare_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.crop_face:
            face, _, _, _ = self._face_detector().crop_face(frame_rgb)
            return np.asarray(face, dtype=np.uint8)
        return cv2.resize(frame_rgb, (self.image_size, self.image_size))

    @staticmethod
    def _target_frame_indices(
        *,
        num_steps: int,
        clip_frames: int,
        frame_count: int,
        fps: float,
        step_seconds: float | None,
        time_offset_seconds: float,
    ) -> np.ndarray:
        if step_seconds is None:
            positions = (
                (np.arange(num_steps)[:, None] + (np.arange(clip_frames) + 0.5) / clip_frames)
                / num_steps
            )
            return np.clip(
                np.rint(positions * (frame_count - 1)).astype(np.int64), 0, frame_count - 1
            )
        if step_seconds <= 0 or fps <= 0:
            raise ValueError("step_seconds and video FPS must be positive")
        seconds = (
            time_offset_seconds
            + np.arange(num_steps)[:, None] * step_seconds
            + (np.arange(clip_frames) + 0.5) * step_seconds / clip_frames
        )
        indices = np.rint(seconds * fps).astype(np.int64)
        invalid = (indices < 0) | (indices >= frame_count)
        indices[invalid] = -1
        return indices

    def _iter_clips(
        self,
        path: Path,
        num_steps: int,
        *,
        step_seconds: float | None = None,
        time_offset_seconds: float = 0.0,
    ) -> Iterator[tuple[int, np.ndarray]]:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            capture.release()
            raise RuntimeError(f"Video reports no decodable frames: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        target_indices = self._target_frame_indices(
            num_steps=num_steps,
            clip_frames=self.clip_frames,
            frame_count=frame_count,
            fps=fps,
            step_seconds=step_seconds,
            time_offset_seconds=time_offset_seconds,
        )
        targets: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for step in range(num_steps):
            if np.any(target_indices[step] < 0):
                continue
            for slot, frame_index in enumerate(target_indices[step]):
                targets[int(frame_index)].append((step, slot))

        frames: list[list[np.ndarray | None]] = [
            [None] * self.clip_frames for _ in range(num_steps)
        ]
        remaining = np.full(num_steps, self.clip_frames, dtype=np.int32)
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                assignments = targets.get(frame_index)
                if assignments:
                    prepared = self._prepare_frame(frame)
                    for step, slot in assignments:
                        frames[step][slot] = prepared
                        remaining[step] -= 1
                    completed_steps = sorted({step for step, _ in assignments if remaining[step] == 0})
                    for step in completed_steps:
                        yield step, np.stack(frames[step])
                        frames[step] = []
                frame_index += 1
        finally:
            capture.release()

    def encode_video(
        self,
        path: str | Path,
        *,
        num_steps: int,
        step_seconds: float | None = None,
        time_offset_seconds: float = 0.0,
    ) -> EncodedSequence:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        video = Path(path).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        features = np.zeros((num_steps, self.output_dim), dtype=np.float32)
        mask = np.zeros(num_steps, dtype=bool)
        if num_steps == 0:
            return EncodedSequence(features=features, mask=mask).validate(
                output_dim=self.output_dim
            )

        import torch

        batch_steps: list[int] = []
        batch_clips: list[np.ndarray] = []

        def flush() -> None:
            if not batch_clips:
                return
            tensor = torch.from_numpy(np.stack(batch_clips)).permute(0, 4, 1, 2, 3)
            tensor = tensor.to(self.device, dtype=torch.float32).div_(255.0)
            amp = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.use_amp and self.device.startswith("cuda")
                else nullcontext()
            )
            with torch.inference_mode(), amp:
                encoded = self.model.extract_features(tensor, keep_seq=False)
            encoded_np = encoded.float().cpu().numpy()
            if encoded_np.shape != (len(batch_steps), self.output_dim):
                raise RuntimeError(
                    f"MARLIN returned {encoded_np.shape}, expected "
                    f"({len(batch_steps)}, {self.output_dim})"
                )
            features[np.asarray(batch_steps)] = encoded_np
            mask[np.asarray(batch_steps)] = True
            batch_steps.clear()
            batch_clips.clear()

        clip_iterator = (
            self._iter_clips(video, num_steps)
            if step_seconds is None
            else self._iter_clips(
                video,
                num_steps,
                step_seconds=step_seconds,
                time_offset_seconds=time_offset_seconds,
            )
        )
        for step, clip in clip_iterator:
            batch_steps.append(step)
            batch_clips.append(clip)
            if len(batch_clips) >= self.batch_size:
                flush()
        flush()
        features[~mask] = 0.0
        return EncodedSequence(features=features, mask=mask).validate(output_dim=self.output_dim)
