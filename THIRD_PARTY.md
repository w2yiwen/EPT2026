# Third-party components

Runtime dependencies are declared in `pyproject.toml` and pinned in
`requirements-lock.txt`. Frozen feature tensors remain external data.

The baseline temporal modules are clean in-repository adaptations of these
official open-source implementations. They reuse EPT-Net's modality encoders,
heads, loss, decoder, and evaluation protocol.

| Baseline | Official source | License | Adapted component |
|---|---|---|---|
| LSTR | https://github.com/amazon-science/long-short-term-transformer | Apache-2.0 | Learned-query long-memory encoder and causal work-memory decoder |
| GateHUB | https://github.com/g1910/GateHUB | MIT | Gated History Unit and causal present decoder; FaH excluded |
| TeSTra | https://github.com/zhaoyue-zephyrus/TeSTra | Apache-2.0 | Exponentially decayed long-memory attention and causal work-memory decoder |
| MulT | https://github.com/yaohungt/Multimodal-Transformer | MIT | All-pairs directional cross-modal Transformers and modality memory encoders |

Copyright and license notices remain with their respective upstream projects.

The optional native-time feature pipeline loads these official pretrained
components. Its adapters call upstream libraries and retain the encoders frozen
in evaluation mode.

| Component | Official source | Use |
|---|---|---|
| MARLIN ViT-Small YTF | https://github.com/ControlNet/MARLIN | 384-dimensional facial clip descriptor; upstream CC BY-NC 4.0 |
| SpeechBrain wav2vec 2.0 IEMOCAP | https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP | 768-dimensional mean-pooled speech descriptor before classification; upstream Apache-2.0 |
| HFL Chinese MacBERT-base | https://huggingface.co/hfl/chinese-macbert-base | 768-dimensional final-layer content-token mean; upstream Apache-2.0 |
| NeuroKit2 0.2.11 | https://github.com/neuropsychology/NeuroKit | Elgendi pulse detection in observed trailing windows; upstream MIT |

Optional dependencies are declared in `pyproject.toml` and
`requirements-paper.txt`. Checkpoint files, model caches and private recordings
remain external to the source repository.
