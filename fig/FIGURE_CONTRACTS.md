# Short-paper figure contracts

These contracts separate scientific evidence from visual rendering. Plotting
scripts do not choose thresholds, repair missing values, smooth curves, or
aggregate raw runs. The `--demo` route exists only to test layout and export.

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
sample should be fixed before viewing this plot—for example, the first eligible
sample in a protocol-defined list or a prespecified median-performance case.
The script never searches for a favorable example.

**Visual encoding.** Coral denotes reference intervals, blue denotes decoded
intervals and event probability, teal/orange denote start/end evidence, and
blue/purple/teal denote EEG-time/EEG-spectrum/PPG reader traces. Grey bands are
non-target intervals. All colors come from the ResearchPilot color library.

## Fig. 5 — Multimodal evidence

**Claim tested.** EEG+PPG physiology and video are complementary if the Full
configuration improves AP and Event mAP and/or reduces Brier relative to both
single-family configurations. The caption must describe the observed result;
the script does not assume that complementarity is present.

**Required matrix.** Exactly nine aggregate cells:

```text
Configurations: Full, EEG+PPG, Video
Metrics:        AP, Brier, Event mAP
```

All cells must come from the same held-out cohort, split manifest, event
decoder, threshold-selection protocol, and seed policy. For three
`aggregate.py` inputs, the script verifies equality of their protocol, data,
calibration-policy, seed records, and source/prepared-data provenance SHA. A
custom combined JSON/CSV must establish all compatibility upstream.

A combined CSV uses one row per cell:

```csv
configuration,metric,mean,lower,upper,n
Full,AP,...,...,...,...
```

`value` may replace `mean`; `lower`/`upper` are optional but must appear
together. A combined JSON may contain `records` with the same fields, or:

```json
{
  "configurations": [
    {
      "name": "Full",
      "metrics": {
        "AP": {"mean": 0.0, "lower": 0.0, "upper": 0.0},
        "Brier": {"mean": 0.0},
        "Event mAP": {"mean": 0.0}
      }
    }
  ]
}
```

The zeros above document structure only and are not experimental values.
Alternatively, pass three `aggregate.py` JSON files as `LABEL=PATH`. Their
`aggregate.frame.average_precision`, `aggregate.frame.brier_score`, and
`aggregate.event.event_map` means are read directly. No uncertainty bars are
drawn for those files unless explicit lower/upper bounds are provided through
the combined format; standard deviation is not silently presented as a
confidence interval. These keys are pooled evaluation metrics, not
`subject_macro.metrics.*` fields, so this panel is a secondary pooled modality
diagnostic.

## Export and QA

Both scripts use exact Arial, the shared color library, fixed physical canvas
sizes, and one Matplotlib canvas for PDF/SVG/PNG. PNG output is 500 dpi. The QA
report records input SHA-256 hashes, dimensions, pixel size, sample/metric
scope, and whether visual preview still requires manual inspection.
