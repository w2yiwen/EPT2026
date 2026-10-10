# Native-time EPT-Net

`configs/paper.yaml` implements the supplied manuscript's textual method under
the existing `eptnet` model name. The package retains `time`, `spec`, `hr`,
`video`, `audio`, `text`, `class_logits`, `boundary_logits` and `offsets` as its
input and output names. The original feature configurations use their original
implementation; `data.input_mode: native_streams` selects this implementation
through `build_model`.

## Install and call

Use Python 3.11 with the repository's pinned PyTorch runtime:

```bash
git clone --branch 1011branch https://github.com/w2yiwen/EPT2026.git
cd EPT2026
python -m pip install -r requirements-paper.txt -e .
```

The entry points share the same configuration:

```bash
ept prepare --config configs/paper.yaml \
  --input data/raw/session_001.pt \
  --output data/processed/bci_subjects_ept_native/sessions/session_001/timeline.pt

ept normalize --config configs/paper.yaml
ept train --config configs/paper.yaml

ept evaluate --config configs/paper.yaml \
  --checkpoint results/1011branch/eptnet/seed_42/best.pt \
  --output results/1011branch/eptnet/seed_42/test_metrics.json

ept infer --checkpoint results/1011branch/eptnet/seed_42/best.pt \
  --input data/processed/bci_subjects_ept_native/sessions/session_001/timeline.pt \
  --output results/1011branch/inference/session_001.jsonl --trace
```

Prepare each recording independently and list its resulting tensor in the
corresponding participant-disjoint manifest. `ept normalize` is optional: training
fits the training statistics when the configured statistics file is absent.
`ept train --resume .../last.pt` restores model, optimizer, epoch, RNG and run
history. Existing output files are preserved by requiring a new destination for
a new run, preparation or inference.

The existing shell entry points also accept `configs/paper.yaml`:

```bash
bash scripts/train.sh configs/paper.yaml
bash scripts/evaluate.sh configs/paper.yaml \
  results/1011branch/eptnet/seed_42/best.pt \
  results/1011branch/eptnet/seed_42
```

## Synchronized recording input

`ept prepare` accepts a PyTorch dictionary saved as `session_001.pt`. Each
timestamp is in seconds. `timeline_origin` is the reference clock value at
session time zero. Device correction follows
`session_time = device_timestamp + time_offsets[device] - timeline_origin`.
Set `time_offsets` from the shared acquisition markers before preparation.

| Field | Shape / content |
|---|---|
| `sample_id`, `subject_id` | Recording ID and participant ID |
| `timeline_origin` | Common reference-clock origin; default 0 |
| `session_start`, `session_end` | Reference-clock recording interval |
| `time_offsets` | Marker corrections for `eeg`, `ppg`, `frame`, `audio` |
| `eeg`, `eeg_times` | `[N,8]` EEG samples at 200 Hz, `[N]` timestamps |
| `ppg`, `ppg_times` | `[N]` PPG samples at 100 Hz, `[N]` timestamps |
| `frames`, `frame_times` | `[N,3,H,W]` RGB facial crops, `[N]` timestamps; uint8 or float in [0,1] |
| `audio`, `audio_times` | `[N]` 16-kHz mono waveform and sample timestamps |
| `words` | List of `{"text": "...", "start": ..., "end": ...}` on the reference clock |
| `events` | Annotated reference-clock `[onset, offset]` pairs; an empty list denotes an annotated negative recording |
| `annotated_segments` | Optional annotated `[start,end]` segments, including target-speaker scope |
| `observation_segments` | Optional observed `[start,end]` segments |

Missing sensors may have empty sample arrays and timestamp arrays. Omit `events`
for an unlabelled recording; preparation then masks supervision. Preparation
does not infer deception annotations from model outputs. `event_targets`
maps annotations to closed decision-grid intervals and merges adjoining
positive steps into maximal runs. Start/end targets and offsets are derived
from those runs, respecting annotation and observation masks.

Facial detection/cropping and acquisition clock markers are supplied with the
recording. The adapter resizes observed crops to 224×224 and chooses 16 evenly
spaced frames from the observed trailing video window. `video_window_seconds`
is an explicit extraction setting; its default is 2 s. Audio uses exactly the
preceding 2 s at 16 kHz. Text uses only words whose completion times are at or
before the current decision, retaining the most recent 128-token context.

## Native tensor and manifest layout

```text
data/processed/bci_subjects_ept_native/
├── normalization_stats.json
├── manifests/
│   ├── sessions_train.jsonl
│   ├── sessions_val.jsonl
│   └── sessions_test.jsonl
└── sessions/
    └── session_001/
        └── timeline.pt
```

Each manifest row names a complete recording. Relative paths resolve from the
manifest's directory:

```json
{"sample_id":"session_001","subject_id":"participant_001","path":"../sessions/session_001/timeline.pt"}
```

Each `timeline.pt` stores a dictionary with `times: [T]` on the 0.5-s grid;
`video: [T,384]`, `audio: [T,768]`, `text: [T,768]`;
`modality_mask: [T,3]`; `sequence_mask: [T]` and `target_mask: [T]`;
`labels: [T]`, `boundaries: [T,2]`, `offsets: [T,2]` and
`positive_mask: [T]`; plus `metadata` and three `streams`:

| Stream | `values` | Representation |
|---|---|---|
| `time` | `[N_time,8]` | Individually filtered EEG samples |
| `spec` | `[N_spec,40]` | Per-channel log band powers every 0.5 s after 4 s of history |
| `hr` | `[N_hr,1]` | Scalar pulse rates at confirmed pulse pairs |

Every stream also stores `support: [N,2]`, `available: [N]` and `mask: [N]`.
Its observations are chronological and availability-ordered. Memory uses the
support endpoint to locate evidence and availability to determine when it can
be consumed. EEG waveform observations remain sample-wise; pulse rates retain
their irregular native sequence. Stable IDs are assigned from the stream's row
indices. `physiology_mask: [T,3]` can explicitly disable a stream at a decision;
otherwise the latest valid representation remains available between updates.

The class encoding remains `0=deception`, `1=truth`. Physiological normalization
uses only valid training observations. The statistics file stores participant
IDs, manifest and recording hashes, means, scales and observation counts; the
checkpoint also carries normalization as model buffers. Validation and test
use those saved buffers.

## Model interface

```python
from eptnet.config import load_config
from eptnet.data.streaming import StreamManifestDataset
from eptnet.models import build_model, load_model

config = load_config("configs/paper.yaml")
model = build_model(config)
session = StreamManifestDataset(config["data"]["train_manifest"])[0]
runtime = model.initial_state()
output = model.step(session.decision(0), runtime)

# The same runtime carries history into the next chronological chunk.
output = model(session, runtime, start=1, stop=min(65, len(session)))
runtime.detach()

# Loading includes the trained physiological normalization.
model = load_model("results/1011branch/eptnet/seed_42/best.pt", device="cpu")
```

`step(decision, runtime)` consumes one decision and mutates only that recording's
runtime. An independent recording gets a fresh `initial_state()`. `forward`
replays precisely this method for chronological chunks. New observations use
`ObservationBatch(values, support, available, indices, mask)` and stable
increasing stream IDs. A re-sent consumed ID creates no duplicate cache entry.

Frozen encoder loading is explicit:

```python
from eptnet.models.pretrained import FrozenBehaviorEncoders

encoders = FrozenBehaviorEncoders(config["encoders"], device="cpu")
```

MARLIN loads through its official `marlin-pytorch` interface; a local
`video_checkpoint` may be configured. SpeechBrain loads the pinned IEMOCAP
snapshot and exposes its finetuned wav2vec2 module before emotion classification.
MacBERT loads its pinned HFL revision. The encoders remain frozen in evaluation
mode. Preparation records the loaded model identities and audiovisual weight
hashes with the recording. Training consumes the resulting frozen descriptors
and jointly learns the projections and downstream modules.

## Manuscript-to-source mapping

| Textual component | Source | Implemented behavior |
|---|---|---|
| III-D: preprocessing | `data/physiology.py` | Stateful second-order high-pass then low-pass filters; EEG 1–45 Hz, PPG 0.5–8 Hz; no reverse filtering |
| III-D: spectral EEG | `CausalPhysiology.decision` | Trailing 4-s Welch, 2-s Hann segments, 50% overlap; [1,4), [4,8), [8,13), [13,30), [30,45] Hz log powers |
| III-D: cardiac features | `CausalPhysiology.decision` | NeuroKit2 0.2.11 Elgendi in 10-s trailing windows; ≥0.4-s confirmation age, ≥0.3-s peak spacing; 60/inter-peak interval bpm; released outputs remain fixed |
| IV-B: frozen behavior features | `models/pretrained.py` | MARLIN token mean + LayerNorm; 768-D wav2vec2 temporal mean; 768-D final MacBERT content-token mean excluding special and padding tokens |
| IV-B: physiological projections | `models/streaming.py` | Training-only standardization and independent input→64→64 GELU MLPs, sample-wise for waveform EEG |
| IV-C / Eq. 1: event guidance | `models/behavior.py` | Masked 4-head audio/video attention; independent AV softmax and boundary sigmoid heads; subsequent text fusion |
| IV-C: HPRM | `models/memory.py` | Timestamp retention, support/availability/ID metadata, latest-input bypass, preceding-decision histories, append once after updating |
| IV-D / Eq. 2: lag and span | `models/ater.py` | Shared two-layer condition MLP, independent affine policies, available-age bounds, span then lag, sigmoid span shift by 2h−1 |
| IV-D / Eq. 3: readout | `models/ater.py` | Four equal-part midpoints, Gaussian width max(span/4,10⁻³), normalized weights over all valid history, one-layer positional Transformer and mean + LayerNorm |
| IV-E / Eq. 4: PMSU | `models/pmsu.py` | Seven separately masked tokens, previous-state attention query, residual feedforward update, GRU candidate, boundary-modulated dimension-wise increment gate |
| IV-E: final heads | `models/streaming.py` | Updated-state event softmax and nonnegative softplus onset/endpoint distances; AV branch supplies boundary responses |
| IV-E: event closure | `evaluation/decoding.py` | Existing causal decoder: highest-confidence offset estimates, rounding, valid-segment clipping, boundary refinement, clipping again; model runtime remains intact |
| IV-F / Eq. 5: objectives | `training/streaming.py` | AV and final CE, start/end BCE, positive-step Smooth-L1; padding and unannotated steps masked |
| IV-F: execution order | `EPTNet.step` | Prune → expose latest observations → AV/context → read preceding history → update state → commit new entries |
| IV-F: TBPTT | `training/streaming.py` | Chronological chunks carry caches and state; detach both at chunk boundaries; reset for independent recordings |
| Evaluation | `evaluation/streaming.py`, existing `evaluation/metrics.py` | Time-step PR/AP and classification, boundary localization and event IoU/F1; per-participant pooling across recordings |

`h = 1-(1-p_start)(1-p_end)` drives both span and state writing. Annotation
boundaries supervise the AV branch; predicted continuous anchors condition
retrieval. Gaussian tails contribute outside the nominal selected interval.
Empty histories produce masked zero evidence, and an update with no valid
tokens retains the state exactly.

## Outputs and configurations

Training writes `resolved_config.yaml`, `run_metadata.json`,
`normalization_stats.json`, `history.json`, `best.pt` and `last.pt` beneath
`results/1011branch/<experiment>/seed_<seed>`. Provenance includes executable
source hashes, participant splits, manifest and prepared recording hashes,
encoder settings and the chronological protocol. New configurations write to
`results/1011branch`.

Prediction JSONL records probabilities, boundary responses, nonnegative offsets,
AV anchors, normalized read lags/spans and dimension-wise write gates.
`read_centers` retains the existing diagnostic name and records each nominal
interval's midpoint in normalized age; `retention_gates` is the effective
previous-state coefficient `1-write_gates` from Eq. 4.
`ept infer --trace` additionally writes each stream's nominal interval in
seconds, lag/span in seconds, observation support/availability times, stable
IDs and the four Gaussian weight vectors. Event JSON retains decoder emission,
peak and closure steps. The decision-grid spacing is recorded as 0.5 s.
Event closure affects only the decoder; the model's state and memories continue.

| Configuration | Change |
|---|---|
| `paper.yaml` | Full native-time EPT-Net |
| `paper_fixed_reader.yaml` | Fixed most-recent normalized span |
| `paper_shared_reader.yaml` | Shared affine lag/span policy across streams |
| `paper_no_event.yaml` | Remove predicted-anchor input and boundary modulation from retrieval/update |
| `paper_no_feedback.yaml` | Remove previous-state input from ATER's condition |
| `paper_no_persistent.yaml` | Initialize the state input to zero at each decision |

The horizon values, video window, dropout, chunk length, optimizer settings and
loss weights are explicit configuration choices. `H_m` is specified symbolically
in the supplied text; the defaults retain 16 s for EEG time and 32 s for spectral
EEG and cardiac history. Changing these experiment settings requires its own
recorded configuration. No evidence-interval annotations are needed for ATER.

This branch was produced by reading and writing source. Training, inference,
experiments and tests were not executed during its preparation.
