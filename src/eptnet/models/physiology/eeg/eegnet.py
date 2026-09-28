"""EPT-facing preprocessing and sequence wrapper for EEGNet-v4."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from torch import Tensor, nn

from .eegnet_v4.vendor.eegnet_core import EEGNetV4


class EEGNetPreprocessor:
    """Convert one raw EEG window to the canonical 8 x 200 float32 contract."""

    def __init__(
        self,
        *,
        n_chans: int = 8,
        target_rate_hz: int = 200,
        n_times: int = 200,
        lowcut_hz: float = 0.5,
        highcut_hz: float = 45.0,
        robust_scale: bool = True,
    ) -> None:
        self.n_chans = int(n_chans)
        self.target_rate_hz = int(target_rate_hz)
        self.n_times = int(n_times)
        self.lowcut_hz = float(lowcut_hz)
        self.highcut_hz = float(highcut_hz)
        self.robust_scale = bool(robust_scale)

    def __call__(self, signal: np.ndarray, *, source_rate_hz: float) -> np.ndarray:
        from scipy import signal as scipy_signal

        values = np.asarray(signal, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != self.n_chans:
            raise ValueError(f"signal must have shape [{self.n_chans}, samples]")
        if not np.isfinite(values).all():
            raise ValueError("EEG window contains NaN or infinity")
        if source_rate_hz <= 2 * self.highcut_hz:
            raise ValueError("source_rate_hz is too low for the configured band-pass")

        values = values - np.median(values, axis=1, keepdims=True)
        sos = scipy_signal.butter(
            4,
            (self.lowcut_hz, self.highcut_hz),
            btype="bandpass",
            fs=float(source_rate_hz),
            output="sos",
        )
        values = scipy_signal.sosfiltfilt(sos, values, axis=1)
        ratio = Fraction(self.target_rate_hz / float(source_rate_hz)).limit_denominator(10_000)
        values = scipy_signal.resample_poly(
            values, ratio.numerator, ratio.denominator, axis=1
        )
        if values.shape[1] < self.n_times:
            raise ValueError(
                f"Resampled EEG has {values.shape[1]} samples; expected at least {self.n_times}"
            )
        values = values[:, : self.n_times]
        if self.robust_scale:
            center = np.median(values, axis=1, keepdims=True)
            mad = np.median(np.abs(values - center), axis=1, keepdims=True)
            scale = 1.4826 * mad
            fallback = values.std(axis=1, keepdims=True)
            scale = np.where(scale > 1e-6, scale, np.maximum(fallback, 1e-6))
            values = (values - center) / scale
        return np.asarray(values, dtype=np.float32)


class EEGNetFeatureEncoder(nn.Module):
    """Apply the same EEGNet weights independently to each aligned raw window."""

    sample_rate_hz = 200

    def __init__(
        self,
        *,
        n_chans: int = 8,
        n_times: int = 200,
        n_outputs: int = 2,
        drop_prob: float = 0.25,
    ) -> None:
        super().__init__()
        self.model = EEGNetV4(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            drop_prob=drop_prob,
        )
        self.n_chans = n_chans
        self.n_times = n_times
        self.output_dim = self.model.feature_dim
        self.n_outputs = n_outputs

    def _flatten_windows(self, eeg: Tensor) -> tuple[Tensor, tuple[int, ...]]:
        if eeg.ndim == 3:
            return eeg, (eeg.shape[0],)
        if eeg.ndim == 4:
            batch, steps, channels, samples = eeg.shape
            return eeg.reshape(batch * steps, channels, samples), (batch, steps)
        raise ValueError("eeg must have shape [B,C,S] or [B,T,C,S]")

    def forward(self, eeg: Tensor) -> Tensor:
        windows, prefix = self._flatten_windows(eeg)
        features = self.model.forward_features(windows)
        return features.reshape(*prefix, self.output_dim)

    def forward_logits(self, eeg: Tensor) -> Tensor:
        windows, prefix = self._flatten_windows(eeg)
        logits = self.model(windows)
        return logits.reshape(*prefix, self.n_outputs)

    def freeze_feature_extractor(self) -> None:
        for parameter in self.model.feature_layers.parameters():
            parameter.requires_grad_(False)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
