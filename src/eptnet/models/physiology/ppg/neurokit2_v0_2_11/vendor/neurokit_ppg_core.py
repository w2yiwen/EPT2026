"""Minimal Elgendi PPG pipeline from NeuroKit2 0.2.11.

Adapted from ``neurokit2.ppg.ppg_clean`` and
``neurokit2.ppg.ppg_findpeaks`` at tag v0.2.11 (commit
86bb55b0062251b1c89e770cb14a56395e0050c1), licensed under MIT.
Only the Butterworth cleaner and Elgendi peak detector needed by the EPT
baseline are retained. Empty-wave handling is made explicit.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal as scipy_signal


def clean_ppg_elgendi(ppg_signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Band-pass a one-dimensional PPG signal at 0.5-8 Hz (Butterworth order 3)."""

    values = np.asarray(ppg_signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("PPG signal must be one-dimensional")
    if len(values) < 16:
        raise ValueError("PPG signal is too short to clean")
    if not np.isfinite(values).all():
        raise ValueError("PPG signal contains NaN or infinity")
    if sampling_rate <= 16:
        raise ValueError("sampling_rate must exceed twice the 8 Hz high cut")
    sos = scipy_signal.butter(
        3,
        (0.5, 8.0),
        btype="bandpass",
        output="sos",
        fs=float(sampling_rate),
    )
    return scipy_signal.sosfiltfilt(sos, values)


def find_peaks_elgendi(
    ppg_cleaned: np.ndarray,
    sampling_rate: float,
    *,
    peakwindow: float = 0.111,
    beatwindow: float = 0.667,
    beatoffset: float = 0.02,
    mindelay: float = 0.3,
) -> np.ndarray:
    """Detect systolic peaks using NeuroKit2's Elgendi implementation."""

    values = np.asarray(ppg_cleaned, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Cleaned PPG must be a finite one-dimensional signal")
    positive = values.copy()
    positive[positive < 0] = 0
    squared = positive**2
    peak_kernel = max(1, int(np.rint(peakwindow * sampling_rate)))
    beat_kernel = max(1, int(np.rint(beatwindow * sampling_rate)))
    moving_peak = ndimage.uniform_filter1d(squared, peak_kernel, mode="nearest")
    moving_beat = ndimage.uniform_filter1d(squared, beat_kernel, mode="nearest")
    threshold = moving_beat + beatoffset * np.mean(squared)

    waves = moving_peak > threshold
    beginnings = np.where(np.logical_and(~waves[:-1], waves[1:]))[0]
    endings = np.where(np.logical_and(waves[:-1], ~waves[1:]))[0]
    if beginnings.size == 0 or endings.size == 0:
        return np.empty(0, dtype=np.int64)
    endings = endings[endings > beginnings[0]]
    if endings.size == 0:
        return np.empty(0, dtype=np.int64)

    minimum_length = int(np.rint(peakwindow * sampling_rate))
    minimum_delay = int(np.rint(mindelay * sampling_rate))
    peaks: list[int] = []
    for beginning, ending in zip(beginnings, endings, strict=False):
        if ending - beginning < minimum_length:
            continue
        local_maxima, properties = scipy_signal.find_peaks(
            values[beginning:ending], prominence=(None, None)
        )
        if local_maxima.size == 0:
            continue
        peak = int(beginning + local_maxima[np.argmax(properties["prominences"])])
        if not peaks or peak - peaks[-1] > minimum_delay:
            peaks.append(peak)
    return np.asarray(peaks, dtype=np.int64)
