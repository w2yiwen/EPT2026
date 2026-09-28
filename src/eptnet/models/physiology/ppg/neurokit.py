"""Causal, fixed-width PPG feature encoder based on NeuroKit2 0.2.11."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

from .neurokit2_v0_2_11.vendor.neurokit_ppg_core import (
    clean_ppg_elgendi,
    find_peaks_elgendi,
)


@dataclass(frozen=True)
class PPGEncodedSequence:
    """Aligned PPG features with both step-level and channel-level availability."""

    features: np.ndarray
    mask: np.ndarray
    channel_mask: np.ndarray

    def validate(self, *, output_dim: int, channels: int) -> PPGEncodedSequence:
        features = np.asarray(self.features)
        mask = np.asarray(self.mask)
        channel_mask = np.asarray(self.channel_mask)
        if features.ndim != 2 or features.shape[1] != output_dim:
            raise ValueError(f"features must have shape [T, {output_dim}]")
        if mask.shape != (features.shape[0],):
            raise ValueError("mask must have shape [T]")
        if channel_mask.shape != (features.shape[0], channels):
            raise ValueError(f"channel_mask must have shape [T, {channels}]")
        if not np.isfinite(features).all():
            raise ValueError("PPG features must be finite")
        if not np.array_equal(mask.astype(bool), channel_mask.astype(bool).any(axis=1)):
            raise ValueError("step mask must equal any(channel_mask)")
        width = output_dim // channels
        for channel in range(channels):
            unavailable = ~channel_mask[:, channel].astype(bool)
            if not np.all(features[unavailable, channel * width : (channel + 1) * width] == 0):
                raise ValueError("unavailable channel features must be exactly zero")
        return self


class NeuroKitPPGEncoder:
    """Extract causal, interpretable features from one or more PPG channels.

    The default backend calls the exact pinned NeuroKit2 package. A minimal
    vendored backend is provided for offline portability and is parity-tested
    against NeuroKit2 0.2.11. Feature aggregation is EPT-specific and never
    uses samples later than the current aligned step.
    """

    source_version = "0.2.11"
    features_per_channel = 20
    feature_names = (
        "clean_mean",
        "clean_std",
        "clean_median",
        "clean_iqr",
        "clean_rms",
        "clean_skewness",
        "clean_excess_kurtosis",
        "peak_count",
        "heart_rate_bpm",
        "ibi_mean_seconds",
        "ibi_std_seconds",
        "rmssd_seconds",
        "pnn50",
        "peak_amplitude_mean",
        "peak_amplitude_std",
        "prominence_mean",
        "prominence_std",
        "width_mean_seconds",
        "width_std_seconds",
        "template_quality",
    )

    def __init__(
        self,
        *,
        channels: int = 2,
        window_seconds: float = 10.0,
        step_seconds: float = 1.0,
        minimum_peaks: int = 4,
        backend: str = "neurokit2",
    ) -> None:
        if channels <= 0 or window_seconds <= 0 or step_seconds <= 0:
            raise ValueError("channels and window durations must be positive")
        if minimum_peaks < 3:
            raise ValueError("minimum_peaks must be at least 3")
        if backend not in {"neurokit2", "vendored"}:
            raise ValueError("backend must be 'neurokit2' or 'vendored'")
        self.channels = int(channels)
        self.window_seconds = float(window_seconds)
        self.step_seconds = float(step_seconds)
        self.minimum_peaks = int(minimum_peaks)
        self.backend = backend
        self.output_dim = self.channels * self.features_per_channel
        self._neurokit = self._load_neurokit() if backend == "neurokit2" else None

    @staticmethod
    def _load_neurokit():
        try:
            import neurokit2
        except ImportError as error:
            raise RuntimeError(
                "NeuroKit2 0.2.11 is required; install the project physiology extra"
            ) from error
        try:
            version = importlib.metadata.version("neurokit2")
        except importlib.metadata.PackageNotFoundError:
            version = getattr(neurokit2, "__version__", "unknown")
        if version != NeuroKitPPGEncoder.source_version:
            raise RuntimeError(
                f"Expected NeuroKit2 {NeuroKitPPGEncoder.source_version}, found {version}"
            )
        return neurokit2

    def _clean_and_find_peaks(
        self, values: np.ndarray, sampling_rate_hz: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._neurokit is None:
            cleaned = clean_ppg_elgendi(values, sampling_rate_hz)
            peaks = find_peaks_elgendi(cleaned, sampling_rate_hz)
        else:
            cleaned = self._neurokit.ppg_clean(
                values, sampling_rate=sampling_rate_hz, method="elgendi"
            )
            peaks = self._neurokit.ppg_findpeaks(
                cleaned, sampling_rate=sampling_rate_hz, method="elgendi"
            )["PPG_Peaks"]
        return np.asarray(cleaned, dtype=np.float64), np.asarray(peaks, dtype=np.int64)

    @staticmethod
    def _template_quality(cleaned: np.ndarray, peaks: np.ndarray) -> float:
        beats = []
        for left, right in zip(peaks[:-1], peaks[1:], strict=False):
            if right - left < 3:
                continue
            beat = scipy_signal.resample(cleaned[left:right], 64)
            beat = (beat - beat.mean()) / max(beat.std(), 1e-8)
            beats.append(beat)
        if len(beats) < 2:
            return 0.0
        matrix = np.stack(beats)
        template = matrix.mean(axis=0)
        correlations = [np.corrcoef(beat, template)[0, 1] for beat in matrix]
        return float(np.clip(np.nanmedian(correlations), -1.0, 1.0))

    def _window_features(
        self, values: np.ndarray, sampling_rate_hz: float
    ) -> np.ndarray | None:
        if len(values) < max(16, int(round(2.0 * sampling_rate_hz))):
            return None
        finite = np.isfinite(values)
        if finite.mean() < 0.95:
            return None
        if not finite.all():
            indices = np.arange(len(values))
            values = np.interp(indices, indices[finite], values[finite])
        if np.ptp(values) <= 1e-8:
            return None
        try:
            cleaned, peaks = self._clean_and_find_peaks(values, sampling_rate_hz)
        except (TypeError, ValueError, IndexError):
            return None
        if len(peaks) < self.minimum_peaks:
            return None

        intervals = np.diff(peaks) / sampling_rate_hz
        differences = np.diff(intervals)
        centered = cleaned - cleaned.mean()
        standard_deviation = max(float(cleaned.std()), 1e-8)
        normalized = centered / standard_deviation
        amplitudes = cleaned[peaks]
        prominences = scipy_signal.peak_prominences(cleaned, peaks)[0]
        widths = scipy_signal.peak_widths(cleaned, peaks, rel_height=0.5)[0] / sampling_rate_hz
        features = np.asarray(
            [
                cleaned.mean(),
                cleaned.std(),
                np.median(cleaned),
                np.quantile(cleaned, 0.75) - np.quantile(cleaned, 0.25),
                np.sqrt(np.mean(cleaned**2)),
                np.mean(normalized**3),
                np.mean(normalized**4) - 3.0,
                len(peaks),
                60.0 / max(intervals.mean(), 1e-8),
                intervals.mean(),
                intervals.std(),
                np.sqrt(np.mean(differences**2)) if differences.size else 0.0,
                np.mean(np.abs(differences) > 0.05) if differences.size else 0.0,
                amplitudes.mean(),
                amplitudes.std(),
                prominences.mean(),
                prominences.std(),
                widths.mean(),
                widths.std(),
                self._template_quality(cleaned, peaks),
            ],
            dtype=np.float64,
        )
        if features.shape != (self.features_per_channel,) or not np.isfinite(features).all():
            return None
        return features.astype(np.float32)

    def encode(
        self,
        waveform: np.ndarray,
        *,
        sampling_rate_hz: float,
        num_steps: int | None = None,
    ) -> PPGEncodedSequence:
        values = np.asarray(waveform, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[0] != self.channels:
            raise ValueError(f"waveform must have shape [{self.channels}, samples]")
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        available_steps = int(np.floor(values.shape[1] / (sampling_rate_hz * self.step_seconds)))
        num_steps = available_steps if num_steps is None else int(num_steps)
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        features = np.zeros((num_steps, self.output_dim), dtype=np.float32)
        channel_mask = np.zeros((num_steps, self.channels), dtype=bool)
        window_samples = int(round(self.window_seconds * sampling_rate_hz))
        step_samples = float(self.step_seconds * sampling_rate_hz)

        for step in range(num_steps):
            end = int(round((step + 1) * step_samples))
            start = end - window_samples
            if start < 0 or end > values.shape[1]:
                continue
            for channel in range(self.channels):
                encoded = self._window_features(values[channel, start:end], sampling_rate_hz)
                if encoded is None:
                    continue
                left = channel * self.features_per_channel
                features[step, left : left + self.features_per_channel] = encoded
                channel_mask[step, channel] = True
        mask = channel_mask.any(axis=1)
        return PPGEncodedSequence(features, mask, channel_mask).validate(
            output_dim=self.output_dim, channels=self.channels
        )
