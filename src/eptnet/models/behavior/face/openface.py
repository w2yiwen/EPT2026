from __future__ import annotations

import csv
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..common import EncodedSequence

OPENFACE_VERSION = "2.2.0"
OPENFACE_SOURCE_COMMIT = "3d4b5cf8d96138be42bed229447f36cbb09a5a29"

AU_INTENSITY_COLUMNS = (
    "AU01_r",
    "AU02_r",
    "AU04_r",
    "AU05_r",
    "AU06_r",
    "AU07_r",
    "AU09_r",
    "AU10_r",
    "AU12_r",
    "AU14_r",
    "AU15_r",
    "AU17_r",
    "AU20_r",
    "AU23_r",
    "AU25_r",
    "AU26_r",
    "AU45_r",
)
AU_PRESENCE_COLUMNS = (
    "AU01_c",
    "AU02_c",
    "AU04_c",
    "AU05_c",
    "AU06_c",
    "AU07_c",
    "AU09_c",
    "AU10_c",
    "AU12_c",
    "AU14_c",
    "AU15_c",
    "AU17_c",
    "AU20_c",
    "AU23_c",
    "AU25_c",
    "AU26_c",
    "AU28_c",
    "AU45_c",
)
POSE_COLUMNS = ("pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz")
GAZE_COLUMNS = ("gaze_angle_x", "gaze_angle_y")
OPENFACE_FRAME_COLUMNS = AU_INTENSITY_COLUMNS + AU_PRESENCE_COLUMNS + POSE_COLUMNS + GAZE_COLUMNS
POOLING_STATISTICS = ("mean", "std", "max", "delta", "mean_abs_velocity")
OPENFACE_FEATURE_NAMES = tuple(
    f"{statistic}_{column}" for statistic in POOLING_STATISTICS for column in OPENFACE_FRAME_COLUMNS
)


def _clean_row(row: Mapping[str, str | None]) -> dict[str, str]:
    return {
        str(key).strip(): str(value).strip()
        for key, value in row.items()
        if key is not None and value is not None
    }


def _float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


class OpenFaceFeatureEncoder:
    """Run official OpenFace and pool interpretable frame outputs into causal bins.

    OpenFace is restricted to academic/non-profit non-commercial research use.
    The runtime is deliberately kept as a local, ignored asset rather than vendored
    as redistributable project source.
    """

    output_dim = len(OPENFACE_FEATURE_NAMES)

    def __init__(
        self,
        *,
        executable: str | Path | None = None,
        min_confidence: float = 0.8,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must lie in [0, 1]")
        self.min_confidence = float(min_confidence)
        self.executable = self._resolve_executable(executable)

    @staticmethod
    def asset_root() -> Path:
        return Path(__file__).resolve().parent / "openface" / "assets"

    @classmethod
    def _resolve_executable(cls, executable: str | Path | None) -> Path | None:
        if executable is not None:
            path = Path(executable).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        candidates = sorted(cls.asset_root().rglob("FeatureExtraction.exe"))
        return candidates[0].resolve() if candidates else None

    def extract_video(self, video_path: str | Path, output_dir: str | Path) -> Path:
        if self.executable is None:
            raise RuntimeError(
                "OpenFace FeatureExtraction.exe is not installed under the face assets directory"
            )
        video = Path(video_path).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        before = {path.resolve() for path in destination.glob("*.csv")}
        command = [
            str(self.executable),
            "-f",
            str(video),
            "-out_dir",
            str(destination),
            "-aus",
            "-pose",
            "-gaze",
            "-q",
        ]
        completed = subprocess.run(
            command,
            cwd=self.executable.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-2000:]
            raise RuntimeError(f"OpenFace failed with exit code {completed.returncode}: {detail}")
        generated = [
            path.resolve() for path in destination.glob("*.csv") if path.resolve() not in before
        ]
        preferred = destination / f"{video.stem}.csv"
        if preferred.resolve() in generated:
            return preferred.resolve()
        if len(generated) != 1:
            raise RuntimeError(f"Expected one OpenFace CSV, found {len(generated)}")
        return generated[0]

    def encode_csv(
        self,
        csv_path: str | Path,
        *,
        num_steps: int | None = None,
        step_seconds: float = 1.0,
        start_seconds: float = 0.0,
    ) -> EncodedSequence:
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        path = Path(csv_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        frames: list[tuple[float, np.ndarray]] = []
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            available = {str(name).strip() for name in (reader.fieldnames or ())}
            required = {"timestamp", *OPENFACE_FRAME_COLUMNS}
            missing = sorted(required - available)
            if missing:
                raise ValueError(f"OpenFace CSV is missing required columns: {missing}")
            for raw_row in reader:
                row = _clean_row(raw_row)
                timestamp = _float(row.get("timestamp"))
                confidence = _float(row.get("confidence"))
                success = _float(row.get("success"))
                if timestamp is None or timestamp < start_seconds:
                    continue
                if confidence is not None and confidence < self.min_confidence:
                    continue
                if success is not None and success < 0.5:
                    continue
                values = [_float(row.get(name)) for name in OPENFACE_FRAME_COLUMNS]
                if any(value is None for value in values):
                    continue
                frames.append((timestamp, np.asarray(values, dtype=np.float64)))

        inferred_steps = (
            max(
                1,
                int(
                    math.floor(
                        (max(timestamp for timestamp, _ in frames) - start_seconds) / step_seconds
                    )
                )
                + 1,
            )
            if frames
            else 0
        )
        steps = inferred_steps if num_steps is None else int(num_steps)
        if steps < 0:
            raise ValueError("num_steps must be non-negative")
        features = np.zeros((steps, self.output_dim), dtype=np.float32)
        mask = np.zeros(steps, dtype=bool)
        grouped: list[list[tuple[float, np.ndarray]]] = [[] for _ in range(steps)]
        for timestamp, values in frames:
            index = int((timestamp - start_seconds) // step_seconds)
            if 0 <= index < steps:
                grouped[index].append((timestamp, values))

        for index, items in enumerate(grouped):
            if not items:
                continue
            items.sort(key=lambda item: item[0])
            times = np.asarray([item[0] for item in items], dtype=np.float64)
            values = np.stack([item[1] for item in items])
            delta = values[-1] - values[0]
            velocity = np.zeros(values.shape[1], dtype=np.float64)
            if len(values) > 1:
                time_delta = np.diff(times)
                valid = time_delta > 0
                if valid.any():
                    velocity = np.mean(
                        np.abs(np.diff(values, axis=0)[valid] / time_delta[valid, None]), axis=0
                    )
            pooled = np.concatenate(
                [values.mean(axis=0), values.std(axis=0), values.max(axis=0), delta, velocity]
            )
            if np.isfinite(pooled).all():
                features[index] = pooled.astype(np.float32)
                mask[index] = True
        features[~mask] = 0.0
        return EncodedSequence(features=features, mask=mask).validate(output_dim=self.output_dim)
