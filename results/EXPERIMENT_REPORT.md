# EPT-Net continuous-session 实验报告

生成时间：`2026-09-27T04:18:55+00:00`  
发布门禁：**PASS（正式矩阵完整）**

## 1. 结论边界

四组三随机种子主实验与三个 seed-42 诊断消融均已通过文件、协议、设备、provenance 和数值一致性校验。下表的 `±` 是 seeds 13/42/73 之间的样本标准差。

即使门禁通过，这仍只是单 participant、单 session 的 within-session pilot。测试集仅 156 个唯一时间步和 30 个目标事件，不足以支持跨受试者、跨 session 或部署泛化结论，也不能单凭三次随机初始化构造跨受试者置信区间。

## 2. 冻结协议

- 训练单位为完整 session：`StitchedManifestDataset` + `SequentialSampler`，每 epoch 对 709 个训练行各计一次，`shuffle=false`、`batch_size=1`；不再使用打乱的重叠窗口优化。
- 验证与测试按 `(session_id, row_index)` 重建唯一连续时间线；测试集固定为 156 步、30 个 deception 事件。
- 正类为 deception（原标签 0）；帧阈值只在验证集网格上选择，然后冻结用于测试集。
- 所有可信主结果和诊断结果禁用 text 输入。旧的 text-enabled 与旧重叠窗口训练结果均不进入本报告。
- `emit_step` 表示因果状态机首次发出在线 alert 的时刻；TTD（步/秒）由该 alert 计算。预测区间只有在后续边界关闭时才 finalized，因此表中 TTD 不是完整区间最终可用时间。
- 各 seed 的 checkpoint/evaluation provenance 必须一致；同一实验的 source/prepared-data fingerprint 必须相同。

## 3. 三随机种子主结果

| 模型 | 状态 | Macro-F1 | Balanced Acc. | AUROC | Boundary F1 | Event F1@0.5 | Event mAP | Alert TTD（步） | Alert TTD（秒） | 分配参数量 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EPT-Net (no text) | complete | 0.4195 ± 0.0560 | 0.5006 ± 0.0788 | 0.5176 ± 0.1237 | 0.3074 ± 0.0572 | 0.0939 ± 0.0911 | 0.0288 ± 0.0316 | 0.0000 ± 0.0000 (n=2; 1 undef.) | 0.1825 ± 0.0389 (n=2; 1 undef.) | 424,396 |
| Early-fusion GRU (no text) | complete | 0.5096 ± 0.0416 | 0.5357 ± 0.0426 | 0.5417 ± 0.0434 | 0.3285 ± 0.0298 | 0.2606 ± 0.0507 | 0.0844 ± 0.0208 | 0.4000 ± 0.1732 | 0.3368 ± 0.1572 | 451,206 |
| Fusion Transformer (no text) | complete | 0.5415 ± 0.0262 | 0.5665 ± 0.0322 | 0.5869 ± 0.0442 | 0.4088 ± 0.0544 | 0.2404 ± 0.1118 | 0.0865 ± 0.0667 | 0.3444 ± 0.1503 | 0.4501 ± 0.0979 | 415,718 |
| EPT-Net w/o recurrent fusion update (no text) | complete | 0.4742 ± 0.0378 | 0.5334 ± 0.0712 | 0.5591 ± 0.0925 | 0.3630 ± 0.0884 | 0.1635 ± 0.0034 | 0.0449 ± 0.0019 | 0.2333 ± 0.2517 | 0.2573 ± 0.0764 | 424,396 |

注：只有 `complete` 行才显示均值 ± 样本 SD；`incomplete` 行故意留空，防止把先完成的 seed 当成正式统计量。
延迟是以成功匹配事件为条件的指标；若某 seed 没有匹配事件，其 TTD 为 `undefined`，汇总会显示有效 n 和缺失数，绝不按 0 填补。

### 所有 seed 的原始测试指标

| 模型 | Seed | 状态 | Macro-F1 | Balanced Acc. | AUROC | Boundary F1 | Event F1@0.5 | Event mAP |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| EPT-Net (no text) | 13 | complete | 0.4330 | 0.5202 | 0.5850 | 0.3386 | 0.1818 | 0.0626 |
| EPT-Net (no text) | 42 | complete | 0.3580 | 0.4139 | 0.3748 | 0.2414 | 0.0000 | 0.0000 |
| EPT-Net (no text) | 73 | complete | 0.4675 | 0.5677 | 0.5930 | 0.3421 | 0.1000 | 0.0238 |
| Early-fusion GRU (no text) | 13 | complete | 0.4806 | 0.5161 | 0.5141 | 0.3131 | 0.3137 | 0.1079 |
| Early-fusion GRU (no text) | 42 | complete | 0.4911 | 0.5065 | 0.5193 | 0.3096 | 0.2128 | 0.0770 |
| Early-fusion GRU (no text) | 73 | complete | 0.5572 | 0.5846 | 0.5916 | 0.3629 | 0.2553 | 0.0683 |
| Fusion Transformer (no text) | 13 | complete | 0.5640 | 0.6030 | 0.6339 | 0.4702 | 0.3448 | 0.1445 |
| Fusion Transformer (no text) | 42 | complete | 0.5127 | 0.5418 | 0.5462 | 0.3895 | 0.1224 | 0.0136 |
| Fusion Transformer (no text) | 73 | complete | 0.5479 | 0.5547 | 0.5805 | 0.3665 | 0.2540 | 0.1013 |
| EPT-Net w/o recurrent fusion update (no text) | 13 | complete | 0.4506 | 0.4541 | 0.4524 | 0.3763 | 0.1639 | 0.0464 |
| EPT-Net w/o recurrent fusion update (no text) | 42 | complete | 0.4542 | 0.5541 | 0.6154 | 0.2687 | 0.1667 | 0.0427 |
| EPT-Net w/o recurrent fusion update (no text) | 73 | complete | 0.5177 | 0.5920 | 0.6096 | 0.4439 | 0.1600 | 0.0456 |

## 4. Seed-42 机制诊断

以下消融只有一个随机种子，不能作为稳定性或显著性结论。完整 EPT-Net 的 seed 42 仅作为同 seed 参考。

| 设定 | 状态 | Macro-F1 | Balanced Acc. | AUROC | Boundary F1 | Event F1@0.5 | Event mAP |
|---|---|---:|---:|---:|---:|---:|---:|
| Full EPT-Net (seed 42) | complete | 0.3580 | 0.4139 | 0.3748 | 0.2414 | 0.0000 | 0.0000 |
| Fixed reader | complete | 0.3910 | 0.4220 | 0.4192 | 0.2985 | 0.1000 | 0.0291 |
| No behavior token | complete | 0.3710 | 0.4742 | 0.4096 | 0.2147 | 0.0606 | 0.0222 |
| EEG-time + behavior only | complete | 0.5121 | 0.5156 | 0.5200 | 0.3816 | 0.1509 | 0.0444 |

## 5. 参数公平性

`allocated` 是模型对象中的全部可训练参数；`executed` 是在完整多任务 loss 反传后实际获得梯度的参数。无文本协议保留但不执行 text projection，因此仅比较 allocated 参数会掩盖架构间的实际执行差异。

| 主模型 | Allocated | Trainable | Executed | Executed / EPT |
|---|---:|---:|---:|---:|
| EPT-Net (no text) | 424,396 | 424,396 | 358,604 | 1.000× |
| Early-fusion GRU (no text) | 451,206 | 451,206 | 358,926 | 1.001× |
| Fusion Transformer (no text) | 415,718 | 415,718 | 348,046 | 0.971× |

`w/o recurrent fusion update` 和三个诊断未单独执行 gradient-reachability 参数审计；其 allocated 参数可在结果 CSV 中查看，但不能把 allocated 相等解释为 executed 容量相等。

## 6. 文本快捷路径隔离与数据限制

审计发现测试集 105/156 个 text 向量与训练集精确重复 (67.3%)；匹配项的训练多数标签一致率为 100.0%，精确查表加训练多数类回退的测试 Balanced Accuracy 为 0.7373。
因此，本报告只接受 experiment 名称以 `continuous_no_text` 表示的冻结结果；text-enabled 结果只可作为泄漏/快捷路径审计对照，不能作为模型性能或模态贡献证据。

全数据审计覆盖 1015 个唯一步、238 个事件，session 数为 1。现有时间切分是同一 participant/session 内的顺序切分，不是 subject-disjoint 或 session-disjoint 测试。上游预计算特征的模型版本、拟合范围和严格因果上下文也不完整，因此最终科学结论仍应标记为 pilot / hypothesis-generating。

## 7. 机器可读产物

- `experiment_summary.csv`：每个正式实验/诊断的完整性门禁与完整实验的均值 ± 样本 SD。
- `experiment_seed_metrics.csv`：每个预期 seed 的原始指标或明确的 `missing_or_invalid`。
- `continuous_report_status.json`：缺失文件、校验错误、provenance 与 release gate 状态。
- 各实验 `aggregate.json` 与 `seed_*/test_metrics.json`：上表的原始来源。

## 8. 最终判断

工程结果门禁：**PASS**。科学证据等级：**受限 pilot**；可以报告本地连续协议结果，但不能宣称已达到跨主体泛化或顶会级实证充分性。
