# Physiological encoder baselines

This package contains two audited, dependency-light physiological baselines:

- `eeg/eegnet.py`: EEGNet-v4 for 8-channel, 200 Hz, one-second raw EEG windows.
  It exposes the 96-dimensional pre-classifier feature and the canonical two-class
  head (1,426 total parameters).
- `ppg/neurokit.py`: NeuroKit2 0.2.11 Elgendi cleaning and peak detection in a
  strict causal 10-second window. It emits 20 interpretable features per channel;
  the current two-channel device therefore produces 40 features.

Third-party core code and licenses are kept inside each model's `vendor/` directory,
with exact source revisions in `model_manifest.json`. The NeuroKit wrapper defaults
to the official pinned package; the vendored core is an explicit offline backend and
is tested for cleaner/peak parity.

Neither encoder reads labels. PPG warm-up and invalid channels are represented by
false masks and exactly-zero feature blocks. Downstream scaling must be fit on train
subjects only. Run `scripts/audit/verify_physiology_encoder_baselines.py` for bounded
forward checks; that script never starts training.

Minimal use:

```python
import numpy as np
import torch

from eptnet.models.physiology import EEGNetFeatureEncoder, NeuroKitPPGEncoder

# One-second raw EEG windows already preprocessed to 200 Hz.
eeg = torch.randn(2, 12, 8, 200)  # [batch, step, channel, sample]
eeg_encoder = EEGNetFeatureEncoder()
eeg_features = eeg_encoder(eeg)               # [2, 12, 96]
eeg_logits = eeg_encoder.forward_logits(eeg)  # [2, 12, 2]

# Two continuous PPG channels at the measured device rate.
ppg = np.random.default_rng(7).normal(size=(2, 385 * 30))
ppg_encoder = NeuroKitPPGEncoder(backend="neurokit2")
encoded = ppg_encoder.encode(ppg, sampling_rate_hz=385)
assert encoded.features.shape == (30, 40)
assert encoded.channel_mask.shape == (30, 2)
```
