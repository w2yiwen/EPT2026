from __future__ import annotations

import importlib.metadata

import numpy as np
import pytest
import torch

from eptnet.models.physiology import (
    EEGNetFeatureEncoder,
    EEGNetPreprocessor,
    NeuroKitPPGEncoder,
)
from eptnet.models.physiology.ppg.neurokit2_v0_2_11.vendor import (
    clean_ppg_elgendi,
    find_peaks_elgendi,
)


def _synthetic_ppg(seconds: int = 14, sampling_rate: int = 385) -> np.ndarray:
    time = np.arange(seconds * sampling_rate, dtype=np.float64) / sampling_rate
    carrier = np.sin(2 * np.pi * 1.15 * time - np.pi / 2)
    morphology = 0.22 * np.sin(2 * np.pi * 2.3 * time - np.pi / 4)
    channel_1 = carrier + morphology + 0.03 * np.sin(2 * np.pi * 0.2 * time)
    channel_2 = 0.9 * carrier + 0.18 * morphology + 0.02 * np.cos(2 * np.pi * 0.3 * time)
    return np.stack((channel_1, channel_2))


def test_eegnet_canonical_contract_and_parameter_count() -> None:
    torch.manual_seed(7)
    encoder = EEGNetFeatureEncoder(drop_prob=0.0).eval()
    eeg = torch.randn(2, 3, 8, 200)
    with torch.inference_mode():
        features = encoder(eeg)
        logits = encoder.forward_logits(eeg)
    assert encoder.output_dim == 96
    assert encoder.parameter_count == 1426
    assert features.shape == (2, 3, 96)
    assert logits.shape == (2, 3, 2)
    assert torch.isfinite(features).all()
    assert torch.isfinite(logits).all()


def test_eegnet_step_encoding_has_no_future_leakage() -> None:
    torch.manual_seed(8)
    encoder = EEGNetFeatureEncoder(drop_prob=0.0).eval()
    eeg = torch.randn(1, 4, 8, 200)
    changed = eeg.clone()
    changed[:, 3] += 1000
    with torch.inference_mode():
        original = encoder(eeg)
        perturbed = encoder(changed)
    torch.testing.assert_close(original[:, :3], perturbed[:, :3], rtol=0, atol=0)


def test_eegnet_preprocessor_resamples_and_forces_float32() -> None:
    rng = np.random.default_rng(9)
    raw = rng.normal(size=(8, 500)).astype(np.float64)
    processed = EEGNetPreprocessor()(raw, source_rate_hz=500.0)
    assert processed.shape == (8, 200)
    assert processed.dtype == np.float32
    assert np.isfinite(processed).all()


def test_vendored_neurokit_encoder_is_causal_and_explicitly_masked() -> None:
    waveform = _synthetic_ppg()
    encoder = NeuroKitPPGEncoder(backend="vendored")
    encoded = encoder.encode(waveform, sampling_rate_hz=385)
    assert encoded.features.shape == (14, 40)
    assert encoded.channel_mask.shape == (14, 2)
    assert not encoded.mask[:9].any()
    assert encoded.mask[9:].all()
    assert np.isfinite(encoded.features).all()
    assert np.all(encoded.features[~encoded.mask] == 0)

    changed = waveform.copy()
    changed[:, -385:] += 1000
    changed_encoded = encoder.encode(changed, sampling_rate_hz=385)
    np.testing.assert_array_equal(encoded.features[:13], changed_encoded.features[:13])
    np.testing.assert_array_equal(encoded.channel_mask[:13], changed_encoded.channel_mask[:13])


def test_neurokit_backend_rejects_unpinned_version(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("neurokit2")
    original = importlib.metadata.version

    def wrong_version(name: str) -> str:
        if name == "neurokit2":
            return "9.9.9"
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", wrong_version)
    with pytest.raises(RuntimeError, match="Expected NeuroKit2 0.2.11"):
        NeuroKitPPGEncoder(backend="neurokit2")


def test_vendored_ppg_core_matches_neurokit_0_2_11() -> None:
    pytest.importorskip("neurokit2")
    if importlib.metadata.version("neurokit2") != "0.2.11":
        pytest.skip("Parity is defined only for NeuroKit2 0.2.11")
    import neurokit2 as nk

    waveform = _synthetic_ppg(seconds=12)[0]
    expected_clean = nk.ppg_clean(waveform, sampling_rate=385, method="elgendi")
    actual_clean = clean_ppg_elgendi(waveform, 385)
    np.testing.assert_allclose(actual_clean, expected_clean, rtol=0, atol=1e-12)
    expected_peaks = nk.ppg_findpeaks(
        expected_clean, sampling_rate=385, method="elgendi"
    )["PPG_Peaks"]
    actual_peaks = find_peaks_elgendi(actual_clean, 385)
    np.testing.assert_array_equal(actual_peaks, expected_peaks)
