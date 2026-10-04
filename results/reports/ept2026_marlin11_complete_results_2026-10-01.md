# EPT2026 Marlin11 正式实验结果报告

报告日期：2026-10-01（UTC）  
范围：冻结的 `bci_subjects_ept_v6_marlin4060_aligned11` 数据集；正式六组实验及图 4、图 5。  
数据来源：各组 `seed_*/test_metrics.json`、`aggregate.json`、`history.json`、`run_metadata.json` 和正式图片 QA 报告。

## 结论摘要

- 六组训练、测试评估和聚合均已完成：EPT-Net、GRU、Transformer、Video-only、Fixed-reader、No-persistent。
- 当前正式矩阵每组只有一个 seed。指标是单次运行结果，不能据此估计跨 seed 方差，也不应将单 seed 差异写成统计显著结论。
- 正式 paper 图 4 和图 5 的 PDF、SVG、500 dpi PNG 均已生成，程序化 QA 检查通过。
- Video-only、Fixed-reader、No-persistent 的评估显式跳过了 checkpoint provenance 总哈希匹配。核对显示它们训练时和评估时的数据工件、split 清单、所用配置和 `src/` 模型源码哈希相同；差异仅在运行脚本、测试文件及已删除的无关 physiology-only 配置。训练哈希和评估哈希都保存在各自 `test_metrics.json` 中。

## 数据与评估范围

冻结数据摘要记录 11 名参与者、4,549 个时间步，时间步长 1 秒。每组测试评估均覆盖相同的 3 条测试序列、1,069 个唯一时间步、645 个 target-valid 时间步和 39 个目标事件。模型输出和帧级评价采用 `positive_class=0`；阈值按 validation 校准。事件指标按 IoU 0.3、0.5、0.7 计算，表中 `Event F1@0.5` 与 `Event mAP` 来自正式测试指标。

训练采用 causal-window 输入，验证使用完整 session 序列；按 participant-mean validation total loss 选择最佳检查点。Video-only 在 epoch 32 达到 patience 早停，Transformer 在 epoch 62 达到 patience 早停；其余训练到 100 epoch。

## 六组结果

| 实验 | Seed | 训练结束 epoch | 最佳 epoch | 最佳验证 loss | 测试阈值 | Frame AP ↑ | Brier ↓ | NLL ↓ | Event mAP ↑ | Event F1@0.5 ↑ | 平均延迟（秒） ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EPT-Net（Full） | 13 | 100 | 100 | 0.4373 | 0.78 | 0.9875 | 0.0297 | 0.1108 | 0.3141 | 0.8608 | 0.9706 |
| GRU | 42 | 100 | 91 | 0.0987 | 0.50 | 1.0000 | 0.0023 | 0.0323 | 0.4140 | 1.0000 | 0.9744 |
| Transformer | 42 | 62（早停） | 47 | 0.7915 | 0.55 | 0.9053 | 0.0707 | 0.2489 | 0.4566 | 0.6602 | 1.5588 |
| Video-only | 42 | 32（早停） | 17 | 2.4642 | 0.61 | 0.6589 | 0.1430 | 0.4441 | 0.1110 | 0.1538 | 1.0000 |
| Fixed-reader | 42 | 100 | 99 | 0.6505 | 0.65 | 0.9713 | 0.0509 | 0.1709 | 0.2735 | 0.8205 | 0.8750 |
| No-persistent | 42 | 100 | 98 | 0.2487 | 0.68 | 0.9824 | 0.0282 | 0.1181 | 0.5856 | 0.8837 | 1.2632 |

聚合文件的 seed 数均为 `n=1`，因此其中显示的 seed 标准差为 0 只是单样本聚合的结果，不代表模型没有随机性。表中延迟是各模型各自 validation 阈值下的结果。

## 结果解读

GRU 在本测试集上的帧 AP、Brier、NLL 和 Event F1@0.5 数值最好；No-persistent 的 Event mAP 最高（0.5856），Transformer 次之（0.4566）。这些是单次运行、不同 seed 的描述性结果，不能单独支持跨 seed 稳定性或因果归因结论。

Full 与 Video-only 的图 5 是用户指定的两组探索性比较，分别使用 seed 13 和 seed 42。图中保留真实 seed 标注；由于 seeds 不匹配，不应将其称为 seed-matched ablation。

## 正式图片

图 4 的测试样本按预先声明规则选择：沿测试清单顺序，取第一个包含至少一个真值目标事件的序列。入选 `session_008_stitched`，包含 14 个真值事件。选择规则及输入哈希保存在 `results/eptnet_marlin11_eeg_ppg_video_no_text/seed_13/fig04_sample_selection.json`。

- 图 4 动态过程图：`fig/fig04_dynamic_tracking/fig04_dynamic_tracking.{pdf,svg,png}`；QA：`fig/fig04_dynamic_tracking/fig04_dynamic_tracking.qa-report.json`。
- 图 5 Full 与 Video 对比：`fig/fig05_modality_evidence/fig05_modality_evidence.{pdf,svg,png}`；QA：`fig/fig05_modality_evidence/fig05_modality_evidence.qa-report.json`。
- 两图尺寸、字体、配色和数据范围程序化检查均通过；QA 仍要求人工目视预览。

## 复现与产物索引

各组结果目录均包含训练 checkpoint、`history.json`、`test_metrics.json`、`test_predictions.jsonl`、`test_events.json` 和 `aggregate.json/csv`：

| 实验 | 聚合结果 |
|---|---|
| EPT-Net | `results/eptnet_marlin11_eeg_ppg_video_no_text/aggregate.json` |
| GRU | `results/gru_marlin11_eeg_ppg_video_no_text/aggregate.json` |
| Transformer | `results/transformer_marlin11_eeg_ppg_video_no_text/aggregate.json` |
| Video-only | `results/eptnet_marlin11_video_only_no_text/aggregate.json` |
| Fixed-reader | `results/eptnet_marlin11_fixed_reader_no_text/aggregate.json` |
| No-persistent | `results/eptnet_marlin11_no_persistent_no_text/aggregate.json` |

`scripts/reporting/build_release_report.py` 针对 18 人、三 seed 的另一套 cohort，不适用于本报告的 Marlin11 数据；本报告直接依据上述 Marlin11 结果产物整理。
