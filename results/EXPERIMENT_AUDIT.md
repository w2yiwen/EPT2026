# EPT-Net 最终实验审计

审计日期：2026-09-27  
总体结论：**工程与本地复现门禁 PASS；论文级科学证据 FAIL；公开发布门禁 WARN**  
结论边界：当前产物可以作为可复核的单被试、单会话 pilot，不足以支持“顶会级泛化”“SOTA”“跨被试有效”或部署鲁棒性声明。

## 冻结协议与证据身份

- 可信实验统一使用 `continuous_session`：训练、验证和测试均由 `StitchedManifestDataset` 重建每个 session 的唯一连续时间线；训练每个 epoch 对 709 个唯一训练行各计一次，`batch_size=1`、顺序采样、无重叠窗重复优化。
- 可信结果统一禁用 text；三种子为 13、42、73；测试集固定为 156 个唯一时间步、30 个正事件。
- 阈值只在 validation 上选择，测试集不参与阈值或超参数选择。
- 当前四个三种子实验共享 provenance SHA-256：`8faff4fba3e30b55ee892c7f0b7ebcf63d3555b856ab393f1da89744376cfdf6`。该身份绑定可执行源码树、全部 manifest 及其有序 tensor 内容、feature schema、normalization statistics、dataset summary 与 dependency lock。
- 正式运行设备记录为 NVIDIA GeForce RTX 4060 Laptop GPU；三个 seed 反映随机初始化波动，不代表被试或会话总体的不确定性。

## 审计矩阵

| 审计项 | 状态 | 证据与判断 |
|---|---|---|
| 本地数据/标签张量契约 | PASS | 标签未平滑或重标；事件边界由原标签生成；切分不截断事件；训练 709 个唯一行、测试 156 个唯一行。此项不替代对上游标注流程的外部核验。 |
| 训练—评估时域一致性 | PASS | 已撤销打乱重叠窗训练，训练/验证/测试均为完整连续 session；persistent state 不再处于“短窗训练、长序列测试”的时域偏移中。 |
| 运行中数据稳定性 | PASS | dataset 在优化前完整 materialize；materialization 前后重算 provenance，源码或 prepared data 发生变化即失败。该保护来自一次真实的训练中 manifest 变化故障。失败运行已隔离，不进入可信结果。 |
| split 与上游预处理隔离 | WARN | 当前代码保证归一化只在 train 拟合、阈值只在 validation 校准；但仅有一个 participant/session，且上游 EEG/视频/音频/text 特征生成代码、模型版本、拟合范围和因果上下文不完整。 |
| 因果性与延迟 | PASS | 三种模型均有 future-perturbation/prefix-invariance 回归测试；Transformer 使用与总序列长度无关的因果正弦位置编码；秒级 TTD 按 `max(word_end(emit)-target_onset, 0)` 计算。 |
| 指标实现 | PASS | deception 正类方向固定；frame、boundary、event AP/mAP、早检和效率指标均有回归测试；event AP 使用按置信度排序的一对一匹配；不可定义延迟保留为缺失值，不按 0 填补。 |
| 配置与运行安全 | PASS | 配置严格拒绝未知/缺失 key 和非法范围；checkpoint schema v3；已有 run 防静默覆盖；resume 拒绝配置、源码或数据身份变化。 |
| 精确恢复训练 | PASS | CPU 集成测试验证 4 epochs 连续训练与 2 epochs + resume 至 4 epochs 的 model、optimizer、scheduler、RNG 与 history bitwise 等价；patience 等关键训练语义不可在 resume 时变化。 |
| provenance 与聚合 | PASS | checkpoint、run metadata、test metrics 和 aggregate 均校验同一 64 位 provenance；缺失或不一致时 fail closed。聚合还校验实验、seed、配置和数据契约。 |
| 测试与静态质量门禁 | PASS | `157 passed`；19 条 warning 均为 PyTorch `norm_first=True` 的 nested-tensor 性能提示；Ruff 与 compileall 通过。 |
| 基线公平性 | PASS/WARN | 实际进入完整多任务反向图的参数量：EPT-Net 358,604；GRU 358,926（+0.09%）；Transformer 348,046（−2.94%）。Transformer 已补因果位置编码；但单数据集调参预算有限，不能据此声称穷尽最强基线。 |
| 统计充分性 | FAIL | 只有 1 participant/session、156 个测试步和 30 个测试正事件；事件高度碎片化，三 seed 不能替代跨被试/跨会话重复。 |
| 核心机制支持 | FAIL | EPT-Net 的三种子 event mAP 为 0.0288±0.0316，低于 GRU 的 0.0844±0.0208 和 Transformer 的 0.0865±0.0667。名为 `no_persistent` 的配置为 0.0449±0.0019，但它只关闭 recurrent fusion update，reader 仍由 `previous_state` 条件化，不能解释为完全移除记忆。 |
| 消融构造有效性 | FAIL | `no_persistent` 不是 memoryless；`single_scale` 同时设置 `use_eeg_spec=false` 与 `use_hr=false`，实际是仅保留 EEG-time + behavior 的混合删减，而不是干净的单一 temporal-scale 对照。现有两项均不得用于单机制因果归因。 |
| 文本启用结果 | QUARANTINED | 105/156 个测试 text 向量与训练集精确匹配，匹配标签一致率 100%；text-enabled 结果和旧重叠窗结果均不得进入主结果、图表或模态贡献结论。 |
| 外部独立复核 | WARN | 本轮为内部代码与实验审计，没有独立机构、独立实现或异构 reviewer backend 背书。 |
| 人类被试衍生产物发布 | WARN | checkpoint、逐步预测、事件轨迹及原始/prepared data 不进入公开白名单；是否进一步共享必须由 consent、伦理审批和数据许可决定。 |

## 三种子结果摘要

以下均为相同连续会话、无文本、三种子协议下的均值 ± 样本标准差：

| 模型 | Macro-F1 | Balanced Acc. | AUROC | Boundary F1 | Event F1@0.5 | Event mAP |
|---|---:|---:|---:|---:|---:|---:|
| EPT-Net | 0.4195±0.0560 | 0.5006±0.0788 | 0.5176±0.1237 | 0.3074±0.0572 | 0.0939±0.0911 | 0.0288±0.0316 |
| Early-fusion GRU | 0.5096±0.0416 | 0.5357±0.0426 | 0.5417±0.0434 | 0.3285±0.0298 | 0.2606±0.0507 | 0.0844±0.0208 |
| Fusion Transformer | 0.5415±0.0262 | 0.5665±0.0322 | 0.5869±0.0442 | 0.4088±0.0544 | 0.2404±0.1118 | 0.0865±0.0667 |
| Config `no_persistent`（仅关闭 recurrent fusion update） | 0.4742±0.0378 | 0.5334±0.0712 | 0.5591±0.0925 | 0.3630±0.0884 | 0.1635±0.0034 | 0.0449±0.0019 |

这些数值是当前 pilot 的如实结果，不支持“EPT-Net 优于强基线”或任何关于 persistent memory 的因果结论；`no_persistent` 只能按其真实干预描述为 **no recurrent fusion update**。`EXPERIMENT_REPORT.md`、各 `aggregate.json` 和 `experiment_seed_metrics.csv` 是表中数值的机器可追踪来源。

## 已完成的高风险修复

1. 用完整、去重、按时间排序的 session 训练替代打乱重叠窗口，消除重复行权重和 persistent horizon 混杂。
2. 在训练、resume、evaluation 和 aggregation 全链路绑定源码与 prepared tensor 内容；中途文件变化、身份缺失或身份不一致均拒绝继续。
3. 引入严格配置 schema、checkpoint schema v3、bitwise resume equivalence 集成测试和运行目录防覆盖。
4. 为 Transformer 加入因果正弦位置编码，并保持截断前缀与完整序列前缀严格一致。
5. 修正秒级 alert TTD 的物理可用时刻；区分首次在线 alert 与最终区间 close。
6. 按实际获得梯度的参数而非仅 allocated 参数匹配主基线；正式矩阵按 config × seed 串行、失败即停并在聚合前验证产物。
7. 默认关闭来源不可审计且存在精确跨 split 重复的 text 分支，将相关历史结果隔离为数据泄漏审计证据。

## 仍然阻断论文级结论的事项

1. 获得足量多 participant、多 session 数据，并采用 subject/session-disjoint 外层测试。
2. 在每个训练折内重建上游特征，公开或冻结 extractor 代码、权重、版本、拟合数据范围与因果上下文。
3. 预注册主指标、阈值选择、事件定义和排除规则；增加跨被试统计、置信区间与适当的配对显著性分析。
4. 解释并修正完整模型弱于强基线的结果；另行设计真正 memoryless 的状态消融，以及一次只改变 temporal scale、模态集合不变的干净对照，再在新数据上重复。
5. 获得独立实现或外部 reviewer 的复核，并补齐代码许可证、引用元数据和 CI 后再宣称公开复现包完整。

## 最终放行判断

- **本地工程执行与证据链：PASS。** 当前结果可复算、可追踪、失败即停，且测试门禁通过。
- **公开代码/汇总指标发布：WARN。** 仅允许白名单中的源码、配置、测试、聚合指标、审计和论文图；不得上传受限数据、checkpoint、逐步预测、事件轨迹或机器元数据。
- **顶会论文性能与泛化主张：FAIL。** 数据规模、切分层级、上游特征 provenance 和当前模型相对性能均不足；代码规范化不能弥补科学证据缺口。
