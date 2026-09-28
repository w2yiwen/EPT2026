# Reproducible experiment figures

The figure generator reads only the complete, frozen 18-subject primary-pilot matrix under `../results`. No score is embedded in plotting code. It validates all four experiments, seeds 13/42/73, run identities, source/data provenance, participant-balanced calibration metadata, subject-macro fields, participant-bootstrap intervals, aggregate files, the de-identified data audit, and parameter-fairness audit before replacing any output.

## Reproduce

From `code/`:

```powershell
python scripts/reporting/gen_fig_experiments.py --check-only
python scripts/reporting/gen_fig_experiments.py --verify-determinism
```

`--check-only` fails closed when the matrix is incomplete or inconsistent. `--verify-determinism` renders twice in isolated temporary directories and requires byte-identical outputs. Successful generation exports vector PDF and 300 dpi PNG, then records generator/source/output SHA-256 hashes and every plotted value in `figure_manifest.json`.

## Statistical encoding

The primary estimator is the mean of per-subject metrics over the four held-out test subjects. Each seed has a fixed-seed, 10,000-resample participant-bootstrap 95% percentile interval. Variation across the three model initializations is displayed separately as sample standard deviation and individual seed markers. Pooled target-step/event metrics use a visually subordinate encoding and are labeled secondary. These quantities answer different questions:

- participant bootstrap: descriptive uncertainty across the four held-out subjects within one model seed;
- seed dispersion: optimization variability;
- pooled score: performance after weighting subjects by their number of target-valid steps/events.

None is a stable population confidence interval with `n=4`. The figures must be captioned as a fixed-split subject-disjoint pilot, not population generalization.

## Figures

- `fig_main_comparison`: subject-macro AUROC, macro-F1, boundary F1, event F1 at IoU 0.5, and event mAP for EPT-Net and the two capacity-matched causal baselines. Participant-bootstrap intervals, model-seed variability, and pooled secondary estimates have distinct encodings.
- `fig_mechanism_ablation`: full EPT-Net versus `no_recurrent_fusion` over all three seeds. The condition still supplies the preceding state to the reader and has fewer executed parameters, so it is labeled a recurrent-fusion-plus-capacity sensitivity rather than a fully memoryless ablation.
- `fig_data_integrity_audit`: the frozen 12/2/4 split, target-valid versus context steps, per-split event/label properties, and heterogeneous modality coverage. It explicitly shows that frozen audio features are unavailable.

All charts use the colorblind-safe Okabe–Ito palette, redundant marker/hatch encodings for grayscale, conference-width typography, deterministic metadata, vector PDF, and 300 dpi PNG.

The older single-session figures and quarantined text-shortcut panels are not part of the current-source primary figure set. They may be discussed only as secondary engineering audits and must not be visually or statistically merged with the 18-subject pilot.
