# EPT2026 partial result recovery

Recovery date: 2026-10-02 (Asia/Shanghai)

Remote project: `/root/EPT2026`

Remote immutable staging copy: `/root/autodl-tmp/ept2026-recovery-20261002-codex`

## Recovery status

- `restored/` contains 19 byte-for-byte copies of files that had previously been downloaded from the server. Their original paths are preserved beneath this directory.
- `evidence/docs_old_snapshot/` contains three additional byte-for-byte copies of the old project documentation downloaded at the same time. They are archived for evidence only and were not copied over the current repository documentation.
- `derived/` contains 12 aggregate files reconstructed deterministically from the six restored `test_metrics.json` files using the repository's `eptnet.aggregate` implementation.
- The reconstructed Full and Video `aggregate.json` files exactly match the SHA-256 values recorded by the original Figure 5 QA report:
  - Full: `8e320750259813d0f8d52236cf441a80836a62d23f8ebc5efb8191b2c513f015`
  - Video: `d5e8903d4c7c4198636abd3ffda1dc5aebc79888f11de6e1c9a93068fc64da8d`
- All restored and reconstructed files were copied into `/root/EPT2026` with no-clobber semantics. No tracked source file was modified; `git status` remained clean because these result artifacts are ignored by Git.

## Important limitation

The six restored `seed_*` directories are partial evidence directories, not complete runnable experiment directories. They contain metrics and metadata, but still lack checkpoints, predictions, event files, and training histories. The formal experiment runner must not treat them as complete, resumable, or safely skippable runs. Before rerunning an experiment, archive or move its recovered partial directory instead of overwriting it.

## Still missing

- All `best.pt` and `last.pt` checkpoints.
- All `history.json`, `test_predictions.jsonl`, and `test_events.json` files.
- Five non-main-model `resolved_config.yaml` files.
- Per-run training CSV/PNG/PDF files and training figure manifests.
- Figure 4 and Figure 5 PDF/SVG originals.
- Figure 4 sample-selection JSON, the frozen sample-id file, and training logs.

Known SHA-256 fingerprints retained for future snapshot recovery:

| Missing artifact | SHA-256 |
|---|---|
| EPT-Net `history.json` | `440a01f7fbed3fd6bc016733fee68482ccc5f4cebb1d7992b6bb5feb603a9a98` |
| GRU `history.json` | `1ff8caabdbbc7b7fb6c6dce97ff3d8334de0d09c87edabfa2d550abb25af9071` |
| Transformer `history.json` | `58025f4edc9ead3b852e450903cbc84ed4e0500de94ee0b68cc5bfbdac0d319f` |
| Video-only `history.json` | `cc947bd85dbc8c8a7f6ae5b439247957d34213ac53a2e667fc2fa5a4ff2e5c80` |
| Fixed-reader `history.json` | `6a02436e0cc55b2ba6aaa30c0c0c4bedbc3fa34a59125c76c78b53482296f143` |
| No-persistent `history.json` | `14515a0cdf5d92c0795137fc21cf80f0d4e76a463dd37a1949da77312b0a2077` |
| EPT-Net `test_predictions.jsonl` | `670ef1edc90322d3a4a481970794da7b7e673f361008f7c9d9e4e763b623113e` |
| EPT-Net `test_events.json` | `44f394c65034601ca5c4d5cc7f6c1f424ddd9a07332bcba60ca81d911e7c6e8d` |

Use `SHA256SUMS` to verify all files in this local recovery bundle.
