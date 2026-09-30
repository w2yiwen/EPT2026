# EPT2026 short-paper blueprint

## 1. Frozen scope and paper identity

This document defines the argument and evidence contract for a 4--6 page
short paper. The executable evidence base is the immutable
`bci_subjects_ept_v6_marlin4060_aligned11` artifact and its frozen train,
validation, and test manifests. It contains the following 11 aligned sessions:

```text
session_002, session_003, session_004, session_006, session_008,
session_009, session_010, session_012, session_015, session_017,
session_018
```

`session_011` is explicitly excluded. Every paper claim, table, figure,
command, and caption remains tied to this frozen executable evidence.

The paper type is **dataset/task-led multimodal BCI with a reference temporal
model and protocol-matched evaluation**. Its scientific identity is the
continuous multimodal task rather than session-level classification.

## 2. Immutable semantics

The repository encoding remains:

```text
0 = deception
1 = truth
positive class = 0
```

The paper-facing quantity is the target-event probability

\[
p_{\mathrm{event}}(t) = P(y_t = 0),
\qquad
z_t = \mathbb{1}[y_t = 0].
\]

After defining this mapping once in the task section, use **target event** and
**reference state** in the scientific narrative. Use *deception* and *truth*
only when documenting the annotation semantics or reproducing a repository
field. Never invert the class probability: every probability, Brier score,
negative log-likelihood, event proposal score, and threshold must use class 0
as positive.

The paper-facing model uses three input families from the frozen artifact:

1. EEG, with time-domain and spectral feature branches;
2. PPG-derived physiology, represented by the implementation's legacy `hr`
   branch name;
3. video-derived behavioral evidence.

Audio and text features may remain stored in the frozen artifact, but the paper
configs disable them. The paper must therefore say **EEG--PPG--video**, not
"all available modalities."

## 3. Central claim

> We formulate multimodal BCI assessment as continuous target-event recognition
> and temporal localization over synchronized EEG, PPG-derived physiology, and
> video. Rather than compressing an interaction into one binary output, the
> proposed reference system produces a causal probability trajectory
> \(p_{\mathrm{event}}(t)\), start/end evidence, and localized event intervals,
> which are evaluated jointly at the probability, frame, boundary, event, and
> detection-delay levels on the frozen aligned multimodal cohort.

This statement describes the implemented task and evaluation interface. Any
comparative performance clause must be completed only after the formal server
runs exist.

### Result sentence to complete after the runs

Use one sentence with only verified values:

> On the frozen subject-disjoint split, EPT-Net achieves **[AP]** frame AP,
> **[Brier]** Brier score, and **[event mAP]** event mAP, while
> **[observed comparison supported by Table 2]**.

Do not fill this sentence from demo figures, smoke runs, a single training log,
or values read from pixels.

## 4. Contribution hierarchy

The paper should present the contributions in this order.

### C1 -- A temporal multimodal BCI task and aligned resource

The primary contribution is an aligned EEG--PPG--video resource with
post-event records mapped to target-event supervision on a shared timeline.
The task asks *when* target-event evidence emerges and *where* an event begins
and ends, rather than asking for one label per interview/session.

Evidence: frozen manifests and metadata, the dataset audit, Table 1, and the
data/annotation portion of Figure 1.

### C2 -- Continuous probabilistic recognition with explicit event boundaries

The task output is a probability trajectory plus start/end and offset outputs.
This supports frame discrimination, proper probability scoring, causal event
decoding, multi-IoU event AP, early recognition, and delay analysis in one
protocol.

Evidence: evaluation contract, Figure 2, and Table 2.

### C3 -- EPT-Net as a compact reference realization

EPT-Net provides modality-specific temporal reading and a persistent causal
state for the task. This is the mechanism used to instantiate the contribution,
not the whole paper. Architecture detail should be limited to what is needed to
understand the probability and boundary outputs.

Evidence: a compact sub-panel in Figure 1, the EPT-Net row in Table 2, and at
most one targeted mechanism diagnostic in the supplement.

### C4 -- An evidence protocol that matches the scientific output

The evaluation keeps three questions separate:

- **discrimination:** frame AP/AUPRC;
- **probability quality:** Brier score and negative log-likelihood;
- **temporal localization:** event AP/mAP at multiple tIoU thresholds,
  boundary F1, event F1, early-detection recall, and detection delay.

Evidence: Table 2, Figure 2, and the optional supplemental reliability plot.

## 5. Claim--evidence ledger

| ID | Paper claim | Current status | Required evidence | Main-paper location |
|---|---|---|---|---|
| CL-01 | The frozen pipeline aligns EEG, PPG-derived physiology, video, and event supervision on one temporal index. | Supported by the frozen data/config contract; numerical metadata must still come from the server artifact. | `dataset_summary.json`, manifests, preflight fingerprints | Fig. 1, Table 1, Task/Data |
| CL-02 | The task output is a time-varying class-0 target-event probability, not a session-level binary decision. | Supported by model/evaluation code. | `positive_probability`, `positive_class=0`, masks | Task, Method, Fig. 2 |
| CL-03 | The system converts continuous evidence into causal localized intervals. | Implemented; empirical quality awaits the formal run. | predictions JSONL, events JSON, event AP/F1/delay | Fig. 2, Table 2 |
| CL-04 | EPT-Net is competitive with causal GRU and Transformer references. | Needs evidence. | declared single-run matrix under the identical frozen protocol, with true seeds reported | Table 2 |
| CL-06 | The reported probability is numerically reliable. | Partially supported only by Brier/NLL implementation; a calibration adjective needs a reliability analysis. | Brier, NLL, optional reliability diagram; validation-only calibration if later added | Table 2, supplement |
| CL-07 | Adaptive reading or persistent state explains the main result. | Optional and needs evidence. | fixed-reader and no-persistent diagnostics | Supplement only |
| CL-08 | The method generalizes to the population or is state of the art. | Outside the current evidence contract. | Requires independent supporting evidence | Do not write |

## 6. Exact experiment and evidence matrix

All commands must read the frozen manifests and write to isolated
`results/*marlin11*_no_text` directories. The maintained orchestrator is:

```bash
bash scripts/experiments/run_marlin11_shortpaper.sh --suite all --device cuda:0
```

The preflight command fingerprints metadata and rejects a changed cohort,
manifest path, label direction, or enabled audio/text branch before training.

| Experiment | Config and formal seeds | Scientific question | Primary outputs | Claim gate |
|---|---|---|---|---|
| Data/protocol audit | `scripts/experiments/preflight_marlin11_shortpaper.py`; no training seed | Is every result tied to the same 11-participant artifact, split, and class mapping? | preflight JSON, metadata hashes, manifest IDs | Mandatory for every figure/table |
| Full EPT-Net | `configs/eptnet_marlin11_eeg_ppg_video.yaml`; 13 | How well does the proposed realization rank frames, score probabilities, and localize events? | per-seed metrics/predictions/events plus `aggregate.json` | Main EPT result |
| Protocol-matched causal GRU | `configs/gru_marlin11_eeg_ppg_video.yaml`; 42 | Does the result require more than conventional early recurrent fusion under the same inputs? | per-seed metrics and aggregate | Table 2 reference |
| Protocol-matched causal Transformer | `configs/transformer_marlin11_eeg_ppg_video.yaml`; 42 | How does EPT compare with causal attention-based fusion? | per-seed metrics and aggregate | Table 2 reference |
| Video only | `configs/eptnet_marlin11_video_only.yaml`; 42 | How does the model behave with behavioral video alone? | per-seed metrics and diagnostic aggregate | Optional input diagnostic |
| Fixed reader | `configs/eptnet_marlin11_fixed_reader.yaml`; 42 | Does adaptive temporal reading add useful behavior under the fixed protocol? | metrics and aggregate | Optional targeted diagnostic |
| No persistent state | `configs/eptnet_marlin11_no_persistent.yaml`; 42 | Does persistent updating add useful behavior under the fixed protocol? | metrics and aggregate | Optional targeted diagnostic |
| Dynamic trace | seed-42 full EPT-Net artifacts; one predeclared held-out `sample_id` | Does the output behave as a temporal probability and localization trace? | paper Figure 2 / repository `fig04` | Qualitative mechanism evidence, not aggregate proof |
| Reliability analysis | full EPT-Net per-step predictions; optional | Are numerical probabilities aligned with empirical frequency? | reliability diagram and bin counts | Supplemental; never replace Brier/NLL |

### Primary metric policy

- Use subject-macro values as the primary aggregation when present; pooled
  values are diagnostics.
- Report frame **AP/AUPRC**, not accuracy, as the main discrimination metric.
  AP has a subject-macro field; AUPRC is currently pooled and must be labelled
  as such.
- Report **Brier score** and **negative log-likelihood** as proper probability
  scores. ECE may be supplemental and must never be the sole probability metric.
- Report event AP at tIoU 0.3, 0.5, and 0.7 plus their mean. Keep this
  threshold-free proposal ranking distinct from thresholded event F1.
- Report the validation-selected operating threshold and detection delay. Test
  labels must never select a threshold.
- The declared seed policy is recorded by the executable runner. Video-only and
  mechanism analyses are single-seed diagnostics.
- Participant-bootstrap intervals describe uncertainty under the held-out
  split. Seed dispersion measures optimization variability; report the two
  quantities separately.

### Decision rules for result language

- **Mechanism evidence:** mention adaptive reading or persistent state as an
  empirical contribution only if the corresponding diagnostic supports it.
  Otherwise keep the mechanism descriptive and move the comparison out of the
  main paper.

## 7. Recommended 4--6 page structure

The budget below assumes references are outside the main-page limit. If the
venue counts references, compress Related Work first, not the task definition
or principal evidence.

| Section | Six-page target | Four-page compression | Responsibility |
|---|---:|---:|---|
| Abstract | 0.25 | 0.20 | Problem, temporal formulation, frozen cohort, method in one clause, verified result, significance |
| 1. Introduction | 0.75 | 0.60 | Establish multimodal BCI need, the clip/session-label gap, the proposed temporal view, and 3 contributions |
| 2. Related Work | 0.45 | folded into Introduction | Multimodal physiological/BCI datasets; online temporal localization; probability assessment |
| 3. Task, Data, and Annotation | 1.10 | 0.75 | EEG--PPG--video alignment, post-event interval supervision, masks, label semantics, frozen split; Fig. 1 and Table 1 |
| 4. Reference Temporal Model | 0.80 | 0.50 | Inputs, modality encoders, adaptive reader/persistent fusion, probability/boundary heads, causal decoding |
| 5. Experiments and Results | 2.10 | 1.55 | Protocol, baselines, metrics, Table 2, Fig. 2, result interpretation |
| 6. Discussion and Conclusion | 0.55 | 0.30 | What the temporal formulation enables and the evidence-supported conclusion |

### Section-level writing plan

#### Abstract

Write last. The first two sentences should establish the scientific problem and
why session-level classification is insufficient. The next two should define
the aligned multimodal resource and temporal output. Use one compact sentence
for EPT-Net. End with verified AP/Brier/event-mAP evidence and the resulting BCI
capability. Do not spend abstract space on implementation history or defensive
caveats.

#### 1. Introduction

1. BCI systems increasingly combine central neural, peripheral physiological,
   and observable behavioral signals, but meaningful states unfold over time.
2. Existing multimodal physiological work largely emphasizes clip-level state
   recognition, while temporal localization work largely studies video alone.
3. Define the paper's target-event probability and interval-localization view.
4. State the four contributions in the hierarchy above, with the model third.

#### 2. Related Work

Use two compact argument groups, not a paper-by-paper list:

- synchronized multimodal BCI/physiological resources and continuous labels;
- online temporal detection, event boundaries, and probability reliability.

The relationship sentence should be: prior work supplies either multimodal
physiology, continuous annotation, or temporal localization machinery; this
paper connects them in one EEG--PPG--video event-recognition protocol.

#### 3. Task, Data, and Annotation

Lead with Figure 1. Define inputs, shared timeline, target-valid mask, interval
targets, and class-0 probability. Report only counts read from the frozen audit.
Explain that invalid/non-target steps may remain causal context but do not enter
supervision or scoring. Table 1 carries the complete dataset/protocol facts so
the prose does not repeatedly foreground cohort size.

#### 4. Reference Temporal Model

Keep this section compact:

1. modality-specific evidence encoders;
2. bounded temporal reading for EEG-time, EEG-spectrum, and PPG;
3. persistent causal multimodal fusion;
4. class, start/end, and offset heads;
5. validation-selected causal event decoder.

Avoid a layer-by-layer inventory unless a detail is needed for reproduction.

#### 5. Experiments and Results

Organize by research question:

- **RQ1:** Can the system jointly recognize and localize target events?
  Answer with Table 2 and Figure 2.
- **RQ2, optional:** Do adaptive reading and persistent state help under the
  same protocol? Answer only in the supplement or one sentence if supported.

Interpret each result immediately after presenting it. Do not narrate the order
in which experiments were attempted.

#### 6. Discussion and Conclusion

Reinforce one memory: multimodal BCI evidence can be represented as a dynamic,
localized probability stream rather than a single interview label. Keep scope
details in Task/Data and end by stating what the aligned temporal formulation
and measured evidence establish.

## 8. Wording guardrails

### Assertive and supportable

- "We formulate the task as continuous target-event recognition and temporal
  localization."
- "We present a synchronized EEG--PPG--video resource with time-localized event
  supervision."
- "EPT-Net produces a class-0 target-event probability at each valid time step
  together with start/end evidence and interval offsets."
- "On the frozen held-out protocol, the full configuration improves ..."
  followed by exact results.

### Prohibited or evidence-dependent

- Do not write "state of the art," "population-level generalization," "robust
  across subjects," or "first" without a separate evidentiary basis.
- Do not call the output "calibrated probability" unless a calibration method
  is fitted on validation data and verified on held-out data. Brier/NLL alone
  support "probability quality," not automatic calibration.
- Do not write "real-time" from causality alone. Use "causal" or "streaming"
  unless measured throughput satisfies a declared real-time requirement.
- Describe deception/truth as the repository annotation classes that define the
  target event; keep the scientific emphasis on multimodal temporal recognition.
- Do not describe Figure 2 as proof of general performance; it is a qualitative
  trace selected by a declared rule.
- State the executable scope precisely in Task/Data, then keep the narrative on
  the temporal task, aligned multimodal resource, and supported findings.

## 9. Verified primary-source anchors

Use these sources for the corresponding claims; consult the paper itself before
final citation insertion.

- Multimodal EEG/PPG/video dataset presentation:
  [Yang et al., Scientific Data 2024](https://www.nature.com/articles/s41597-024-03676-4).
- BCI multimodal synchronization and a shared time base:
  [Guttmann-Flury et al., Scientific Data 2025](https://www.nature.com/articles/s41597-025-04861-9).
- Recent brain--body--video dataset validation:
  [Kashevnik et al., Scientific Data 2026](https://www.nature.com/articles/s41597-026-07209-z).
- Continuous affect trajectories with physiology:
  [Sharma et al., Scientific Data 2019](https://www.nature.com/articles/s41597-019-0209-0).
- EEG representation and BCI task framing:
  [EEGPT, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html).
- Dataset-scale temporal interval annotations:
  [EgoExoLearn, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_EgoExoLearn_A_Dataset_for_Bridging_Asynchronous_Ego-_and_Exo-centric_View_CVPR_2024_paper.html).
- Per-time scores and boundary regression:
  [ActionFormer, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7278_ECCV_2022_paper.php).
- Relative boundary evidence and multi-tIoU evaluation:
  [TriDet, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Shi_TriDet_Temporal_Action_Detection_With_Relative_Boundary_Modeling_CVPR_2023_paper.html).
- Progress-aware temporal traces:
  [Progress-Aware Online Action Segmentation, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.html).
- Joint frame/event/latency reporting:
  [Context-Enhanced Memory-Refined Transformer, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Pang_Context-Enhanced_Memory-Refined_Transformer_for_Online_Action_Detection_CVPR_2025_paper.html).
- Reliability diagrams and temperature scaling:
  [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html).
- Why ECE should not be reported alone:
  [Chidambaram et al., ICML 2024](https://proceedings.mlr.press/v235/chidambaram24a.html).
