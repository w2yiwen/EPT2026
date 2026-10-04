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
