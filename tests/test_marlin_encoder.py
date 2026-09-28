from __future__ import annotations

from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

from eptnet.models.behavior.face.marlin import MARLIN_OUTPUT_DIM, MarlinFeatureEncoder


class _FakeMarlin(torch.nn.Module):
    def extract_features(self, clips: torch.Tensor, *, keep_seq: bool) -> torch.Tensor:
        assert clips.shape[1:] == (3, 16, 224, 224)
        assert keep_seq is False
        means = clips.mean(dim=(1, 2, 3, 4), keepdim=False)
        return means[:, None].repeat(1, MARLIN_OUTPUT_DIM)


class _FakeFaceDetector:
    def __init__(self) -> None:
        self.calls = 0

    def crop_face(self, _frame: np.ndarray):
        self.calls += 1
        return np.zeros((224, 224, 3), dtype=np.uint8), 1, 0, 0


def test_marlin_encoder_preserves_step_contract(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    encoder = MarlinFeatureEncoder(
        device="cpu",
        batch_size=2,
        crop_face=False,
        use_amp=False,
        model=_FakeMarlin(),
    )

    def clips(
        self: MarlinFeatureEncoder,
        path: Path,
        num_steps: int,
        **_kwargs: object,
    ):
        assert path == video.resolve()
        assert num_steps == 3
        yield 0, np.zeros((16, 224, 224, 3), dtype=np.uint8)
        yield 2, np.full((16, 224, 224, 3), 255, dtype=np.uint8)

    encoder._iter_clips = MethodType(clips, encoder)
    encoded = encoder.encode_video(video, num_steps=3)

    assert encoded.features.shape == (3, MARLIN_OUTPUT_DIM)
    assert encoded.mask.tolist() == [True, False, True]
    assert np.all(encoded.features[0] == 0.0)
    assert np.all(encoded.features[1] == 0.0)
    assert np.allclose(encoded.features[2], 1.0)


def test_marlin_detects_each_sampled_frame_independently() -> None:
    pytest.importorskip("cv2")
    encoder = MarlinFeatureEncoder(
        device="cpu",
        crop_face=True,
        use_amp=False,
        model=_FakeMarlin(),
    )
    detector = _FakeFaceDetector()
    encoder._face_detector = MethodType(lambda _self: detector, encoder)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    prepared = np.stack([encoder._prepare_frame(frame) for _ in range(16)])

    assert detector.calls == 16
    assert prepared.shape == (16, 224, 224, 3)
    assert prepared.dtype == np.uint8


def test_marlin_real_time_indices_apply_video_audio_offset() -> None:
    indices = MarlinFeatureEncoder._target_frame_indices(
        num_steps=2,
        clip_frames=2,
        frame_count=100,
        fps=10.0,
        step_seconds=1.0,
        time_offset_seconds=2.0,
    )

    assert indices.tolist() == [[22, 28], [32, 38]]


def test_marlin_real_time_indices_fail_closed_outside_video() -> None:
    indices = MarlinFeatureEncoder._target_frame_indices(
        num_steps=2,
        clip_frames=2,
        frame_count=10,
        fps=10.0,
        step_seconds=1.0,
        time_offset_seconds=-1.0,
    )

    assert indices[0].tolist() == [-1, -1]
