# EPT-Net

EPT-Net is a causal multimodal network for continuous event recognition and temporal localization. The implementation combines EEG, PPG-derived physiology, and video features with modality-specific memory, adaptive reading, and a persistent temporal state.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt -e .
```

## Data

Place the prepared feature dataset at:

```text
data/processed/bci_subjects_ept_v6_aligned11/
├── dataset_summary.json
├── feature_schema.json
├── normalization_stats.npz
├── manifests/
└── sessions/
```

## Usage

Train one model:

```bash
bash scripts/train.sh configs/main.yaml
```

Evaluate one checkpoint:

```bash
bash scripts/evaluate.sh \
  configs/main.yaml \
  results/final/eptnet/seed_42/best.pt \
  results/final/eptnet/seed_42
```

Run the complete experiment set and generate PNG figures:

```bash
bash scripts/run_experiments.sh
```

Generate figures from existing results:

```bash
python figures/generate.py
```

## Structure

```text
configs/       model and ablation configurations
figures/       PNG figure generator and generated images
scripts/       training and evaluation entry points
src/eptnet/    model, data, training, and evaluation code
results/final/ completed runs, metrics, predictions, and checkpoints
```

Available configurations are `main`, `fixed_reader`, `no_persistent`, `video_only`, `lstr`, `gatehub`, and `testra`.

Each completed run stores its resolved configuration, metadata, training history, `best.pt`, `last.pt`, test metrics, event predictions, and frame-level predictions. The figure generator writes PNG files only.
