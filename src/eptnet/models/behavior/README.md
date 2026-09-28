# Behavior encoder baselines

This package keeps the three audited behavior encoders under `models/`, separated
by modality:

- `face/openface`: OpenFace 2.2.0 runtime wrapper and 215-dimensional interpretable
  AU/presence/pose/gaze statistics;
- `audio/wavlm`: frozen Microsoft WavLM Base+ with one-second causal pooling;
- `text/macbert`: frozen HFL Chinese MacBERT-base with masked-mean pooling;
- `temporal/causal_tcn`: the same two-layer 128-dimensional causal decoder for
  every single-modality comparison.

Downloaded weights and executables live in each encoder's `assets/` directory.
They are ignored because they are large and, for OpenFace, redistribution is
license-restricted. Run `scripts/data/install_behavior_encoder_assets.py` to
recreate the local asset layout. The installer also executes OpenFace's official
second-stage model downloader and verifies the four CEN patch-model sizes and hashes;
an interrupted or mismatched download is replaced only inside this generated asset
directory.

The public feature contract is `EncodedSequence(features=[T,D], mask=[T])`. Empty or
invalid steps are exactly zero with a false mask. Text is encoded independently per
observed step, audio is split into independent causal one-second chunks, and face
statistics never cross their one-second bins. The shared TCN is prefix-causal; the
model-selection split and downstream loss/metric masking remain the caller's
responsibility.

OpenFace is restricted to academic/non-profit non-commercial research. WavLM
source is MIT-licensed. MacBERT source and the published Hugging Face model card
state Apache-2.0. See each `model_manifest.json` before redistribution.
