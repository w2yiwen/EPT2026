# EPT-Net 发布实验报告

产物门禁：**INCOMPLETE（不得引用为最终结果）**
科学证据等级：**固定 subject-disjoint split 的 pilot**；不是交叉验证，也不是人口级泛化证据。

## 1. 证据边界与门禁

正式矩阵尚未完整。任何不完整实验的局部均值均被抑制；补齐并重新运行构建器前，不得据此比较模型、宣称机制有效或声称达到顶会实证标准。

### 缺失或无效产物

- `eptnet_bci_subjects_no_text`：缺少 `eptnet_bci_subjects_no_text/aggregate.json`
- `eptnet_bci_subjects_no_text`：缺少 `eptnet_bci_subjects_no_text/seed_13/test_metrics.json`
- `eptnet_bci_subjects_no_text`：缺少 `eptnet_bci_subjects_no_text/seed_42/test_metrics.json`
- `eptnet_bci_subjects_no_text`：缺少 `eptnet_bci_subjects_no_text/seed_73/test_metrics.json`
- `baseline_early_fusion_gru_bci_subjects_no_text`：缺少 `baseline_early_fusion_gru_bci_subjects_no_text/aggregate.json`
- `baseline_early_fusion_gru_bci_subjects_no_text`：缺少 `baseline_early_fusion_gru_bci_subjects_no_text/seed_13/test_metrics.json`
- `baseline_early_fusion_gru_bci_subjects_no_text`：缺少 `baseline_early_fusion_gru_bci_subjects_no_text/seed_42/test_metrics.json`
- `baseline_early_fusion_gru_bci_subjects_no_text`：缺少 `baseline_early_fusion_gru_bci_subjects_no_text/seed_73/test_metrics.json`
- `baseline_fusion_transformer_bci_subjects_no_text`：缺少 `baseline_fusion_transformer_bci_subjects_no_text/aggregate.json`
- `baseline_fusion_transformer_bci_subjects_no_text`：缺少 `baseline_fusion_transformer_bci_subjects_no_text/seed_13/test_metrics.json`
- `baseline_fusion_transformer_bci_subjects_no_text`：缺少 `baseline_fusion_transformer_bci_subjects_no_text/seed_42/test_metrics.json`
- `baseline_fusion_transformer_bci_subjects_no_text`：缺少 `baseline_fusion_transformer_bci_subjects_no_text/seed_73/test_metrics.json`
- `ablation_no_recurrent_fusion_bci_subjects_no_text`：缺少 `ablation_no_recurrent_fusion_bci_subjects_no_text/aggregate.json`
- `ablation_no_recurrent_fusion_bci_subjects_no_text`：缺少 `ablation_no_recurrent_fusion_bci_subjects_no_text/seed_13/test_metrics.json`
- `ablation_no_recurrent_fusion_bci_subjects_no_text`：缺少 `ablation_no_recurrent_fusion_bci_subjects_no_text/seed_42/test_metrics.json`
- `ablation_no_recurrent_fusion_bci_subjects_no_text`：缺少 `ablation_no_recurrent_fusion_bci_subjects_no_text/seed_73/test_metrics.json`

本公开 release 仅包含 primary 4×3 矩阵。历史单会话工程结果不进入 release readiness，也不进入公开机器可读数值。产物门禁 PASS 只表示本地冻结协议可追溯，不等于科学结论或顶会证据充分。

## 2. Primary：18-subject fixed-split pilot

- 训练前冻结、subject-disjoint 的 12/2/4 train/validation/test 划分；这是 label-aware、modality-constrained stratification，且仅有一个 fixed split。
- 模型观察完整交互上下文，但 loss、阈值选择和测试指标仅在 target-speaker mask 内计算；无效说话人区间会关闭且不能开启预测事件。
- text 与 audio 输入禁用。阈值只依据 2 个 validation subjects 的 participant-mean Macro-F1 在 0.10–0.90 的 81 点网格上选择，随后冻结到 test。
- Event F1、early detection 与 latency 使用冻结帧阈值的因果 decoder；event AP/mAP 则对每个 target-valid step 生成一个由正类概率评分、offset 定界的稠密候选，按连续 target segment 截断，不使用 score threshold 或 NMS。因此 mAP 不随 validation-selected / fixed-0.5 operating point 改变。
- 每个 seed 的 participant-macro 指标附 10,000 次 participant bootstrap 区间；下表的 `±` 则是三个模型初始化 seed 之间的样本标准差。两者都不能替代 repeated group splits。

冻结数据计数：18 subjects、7405 context steps、5490 target-valid steps；test 为 4 subjects / 1505 context / 1201 target-valid steps / 55 events。

### Participant-macro 主结果（validation-selected threshold）

| 模型 | 状态 | Frame Macro-F1 | Balanced Acc. | AUROC | AP | Boundary F1 | Event F1@0.5 | Threshold-free Event mAP | TTD seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EPT-Net | incomplete | — | — | — | — | — | — | — | — |
| Early-fusion GRU | incomplete | — | — | — | — | — | — | — | — |
| Fusion Transformer | incomplete | — | — | — | — | — | — | — | — |
| EPT-Net w/o recurrent fusion update | incomplete | — | — | — | — | — | — | — | — |

TTD 是成功匹配事件条件下的指标。没有匹配事件的 seed 保持 `undefined`，汇总明确显示有效 seed n 与 undefined 数，绝不补成 0。participant-level 的 n、missing 和 bootstrap CI 保存在 `release_seed_metrics.csv`。

### 固定阈值 0.5 敏感性分析

该敏感性分析只改变 operating-point 指标。Threshold-free event AP/mAP 使用相同稠密候选排名，构建器会逐 seed 强制验证 selected 与 fixed-0.5 的 AP@各 IoU 和 mAP 完全一致。

| 模型 | Selected Frame Macro-F1 | Fixed-0.5 Frame Macro-F1 | Selected Event F1@0.5 | Fixed-0.5 Event F1@0.5 |
|---|---:|---:|---:|---:|
| EPT-Net | — | — | — | — |
| Early-fusion GRU | — | — | — | — |
| Fusion Transformer | — | — | — | — |
| EPT-Net w/o recurrent fusion update | — | — | — | — |

### 参数公平性

`allocated` 是模型对象中全部可训练参数；`executed` 是完整多任务 loss 反传后实际收到梯度的参数。baseline 比较按 executed 参数控制在 EPT-Net 的 ±3% 内。

| 模型 | Allocated | Trainable | Executed | Executed / EPT |
|---|---:|---:|---:|---:|
| EPT-Net | 403,852 | 403,852 | 334,796 | 1.000× |
| Early-fusion GRU | 436,362 | 436,362 | 334,682 | 1.000× |
| Fusion Transformer | 419,066 | 419,066 | 343,626 | 1.026× |
| EPT-Net w/o recurrent fusion update | 403,852 | 403,852 | 276,428 | 0.826× |

`w/o recurrent fusion update` 仅执行 EPT-Net 的 82.6% 参数，因此是容量混杂的机制消融，不能把差异单独归因于 recurrent fusion update。

## 3. 可追溯性与发布判断

- 每个 seed 的 checkpoint、evaluation 与 run metadata provenance 必须相同；primary 矩阵的所有可用实验必须共享同一 provenance。
- Primary evaluation provenance 中的 dataset summary、feature schema、normalization statistics 和 train/val/test manifest+tensors 摘要必须逐项匹配匿名数据审计。
- `aggregate.json` 必须与三份 seed metrics 在 identity、数值、样本标准差、n/missing 和conditional latency 上逐项一致。
- `release_summary.csv` 提供四模型的三-seed 汇总；`release_seed_metrics.csv` 保留原始 seed 值、participant bootstrap、校准阈值和 provenance。
- 报告不含生成时间，JSON key、行顺序与 LF 换行固定，输入不变时输出字节稳定。

工程/产物门禁：**INCOMPLETE**。在缺失项补齐前，不输出模型优劣、机制有效性、统计显著性或顶会性能结论。
