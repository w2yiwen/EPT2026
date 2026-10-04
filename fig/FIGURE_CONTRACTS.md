# Short-paper figure contracts

These contracts separate scientific evidence from visual rendering. Plotting
scripts do not choose thresholds, repair missing values, smooth curves, or
aggregate raw runs. The `--demo` route exists only to test layout and export.

## Fig. 3 — Seed-matched main comparison

The main figure compares exactly EPT-Net, LSTR, causal GateHUB, and TeSTra on
frame AP, Brier score, event F1 at tIoU 0.5, and event mAP. Each input is the
formal aggregate for seed 42. The renderer requires identical data,
evaluation, calibration, and provenance metadata and performs no additional
aggregation. EPT-Net is the sole accent color; baselines remain neutral.

## Fig. 4 — Dynamic tracking and temporal localization

**Claim tested.** The system can continuously track target-event evidence and
convert that evidence into localized intervals, while its physiological reader
adapts the temporal context used by EEG and PPG branches.

**Required sources.**

1. `evaluate.py` `*_predictions.jsonl`, containing one row per sequence step.
2. The matching `*_events.json`, whose sequence lists use the same first-
   appearance sample order as the JSONL file.
3. The matching `test_metrics.json`, used only for the validation-selected
   frame threshold and the fixed boundary threshold. An explicit
   `--frame-threshold` is permitted when the threshold provenance is recorded
   separately.

The renderer validates required fields and positional sequence alignment, but
does not compare run or provenance fingerprints across these three files. Their
same-run identity must be established upstream.

Each selected JSONL row must contain:

```text
sample_id, row_index, positive_class, target_valid, positive_probability,
boundary_probabilities[2], read_centers[3], read_widths[3]
```

The renderer fails unless `positive_class` is exactly `0`, preserving the
repository-wide `0=deception, 1=truth` convention. The plotted curve is always
\(p_{event}(t)=P(y_t=0)\), never a silently inverted class-1 score.

`start_time_seconds` and `end_time_seconds` are optional, but must be present
for every selected step if used. The physiological-reader branch order follows
the model implementation: EEG time, EEG spectrum, PPG-derived physiology
(internally named `hr`). Centers and widths are normalized cache coordinates.

**Selection safeguard.** `--sample-id` is mandatory for formal rendering. The
evaluation-view sample should be fixed before viewing this plot—for example,
the first eligible sample in a protocol-defined list. Under the current all-11
training policy it is not a held-out sample.
The script never searches for a favorable example.

**Visual encoding.** Coral denotes reference intervals, blue denotes decoded
intervals and event probability, teal/orange denote start/end evidence, and
blue/purple/teal denote EEG-time/EEG-spectrum/PPG reader traces. Grey bands are
non-target intervals. All colors come from the ResearchPilot color library.

## Export and QA

The formal script uses exact Arial, the shared color library, fixed physical canvas
sizes, and one Matplotlib canvas for PDF/SVG/PNG. PNG output is 500 dpi. The QA
report records input SHA-256 hashes, dimensions, pixel size, sample/metric
scope, and whether visual preview still requires manual inspection.

## Fig. 5 — Full versus Video exploratory evidence

The figure contains exactly two configurations, Full and Video, and reports AP,
Brier, and Event mAP from their upstream aggregates. The two aggregates may use
different seeds. Their actual seed lists must be read from aggregate metadata,
displayed in the figure, and recorded in the QA report; seed identities must
never be rewritten. Protocol, data scope, calibration policy, and provenance
must still match. Because the seed policy is unmatched, this figure is an
exploratory comparison rather than a seed-matched ablation.
