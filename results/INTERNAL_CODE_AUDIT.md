# EPT-Net 内部复现级代码审计

审计日期：2026-09-27  
范围：`code/src`、`code/configs`、`code/scripts`、`code/tests`、正式结果协议与公开发布边界  
审计性质：内部代码与实验审计；当前没有独立机构、独立实现或异构 reviewer backend，因此不能作为外部独立背书。

## 总体结论

- 工程实现与本地复现：**PASS**。可信协议已统一为完整连续 session；严格配置、内容级 provenance、因果 Transformer、物理延迟、精确 resume、正式矩阵运行器和聚合门禁均已落地。`157 passed`，Ruff 与 compileall 通过。
- 论文性能与泛化证据：**FAIL / 暂不放行**。只有一个 participant/session、156 个测试时间步与 30 个正事件；上游特征提取 provenance 不完整；EPT-Net 未超过两个强时序基线；现有 `no_persistent` 与 `single_scale` 配置还存在构造混杂，不能支持机制归因。
- 公开发布：**WARN / 仅白名单放行**。可以发布源码、配置、测试、聚合结果、审计和论文图；不得发布原始/prepared 被试数据、checkpoint、逐步预测、事件轨迹、失败运行或机器特定元数据。数据衍生物的进一步共享必须由 consent、伦理审批与数据许可决定。

## 当前冻结证据

| 项目 | 冻结值 |
|---|---|
| 训练协议 | `continuous_session`；完整去重 session；`batch_size=1`；顺序采样；无 shuffle |
| 每 epoch 训练行 | 709 个唯一时间步，各计一次 |
| 测试规模 | 1 个 session；156 个唯一时间步；30 个正事件 |
| 可信种子 | 13、42、73 |
| 可信输入 | text disabled |
| provenance SHA-256 | `8faff4fba3e30b55ee892c7f0b7ebcf63d3555b856ab393f1da89744376cfdf6` |
| checkpoint schema | v3 |
| 本地测试 | 157 passed；19 个已知非失败 warning |
| 主结果状态 | EPT-Net、GRU、Transformer，以及配置名为 `no_persistent` 的 no-recurrent-fusion-update 对照，四个三种子 aggregate 完整 |

## 发现与处置状态

### A01 — OPEN / BLOCKER：现有数据不能支持顶会级泛化结论

证据：`data_audit.json` 和 `EXPERIMENT_REPORT.md` 记录全数据仅一个 participant/session；测试仅 156 个唯一时间步和 30 个 deception 事件。现有顺序切分是 within-session evaluation，不是 subject-disjoint 或 session-disjoint benchmark。标签事件高度碎片化，三个随机种子只量化初始化波动。

影响：所有 SOTA、跨人群泛化、部署鲁棒性与人群统计显著性表述均不成立。工程质量不能补偿样本层级和外部效度不足。

关闭条件：获取足量多 participant、多 session 数据；冻结 subject/session-disjoint 外层测试；预注册主指标、事件定义、排除规则和统计检验；在独立测试上复验。

### A02 — CLOSED：persistent state 的训练/评估时域已对齐

旧问题：训练曾使用独立、打乱的重叠窗口，而验证/测试使用整段 session，导致 persistent state 在短窗上训练、长序列上测试。

处置：`train.py` 的可信入口现在强制 `training.sequence_protocol=continuous_session` 和 `batch_size=1`，通过 `StitchedManifestDataset` 对训练、验证和测试重建完整、唯一、连续的 session。run metadata 冻结 dataset adapter、完整 session 单位、顺序 sampler、`shuffle=false` 与 batch 数。配置和 resume 均拒绝协议变化。

验证：`test_dataset_sequence.py`、`test_engine_reproducibility.py` 和真实数据契约测试覆盖 stitching、去重、连续 row、loader 协议与 metadata。完整模型、两个基线和名为 `no_persistent` 的配置已按新协议重跑三种子；该配置的真实干预边界见 A15。

### A03 — CLOSED：秒级 TTD 使用真实物理可用时刻

秒级 alert 延迟统一为：

```text
max(word_end(emit_step) - target_onset, 0)
```

即使发射词与事件起点重叠，也不会因 `emit_step <= target_start` 被错误强制为零。步级 TTD 和原始 `emit_step` 保留；无成功匹配事件的 seed 将延迟记为未定义，聚合报告有效 n 与缺失数而不是填 0。回归测试覆盖重叠词区间。

### A04 — CLOSED：结果已绑定源码与 prepared tensor 内容

`provenance.py` 生成可移植内容身份，覆盖：

- allowlist 内的可执行源码、配置、脚本、测试与环境声明；
- 每个 manifest 的原始内容及按 manifest 顺序组合的全部 tensor 内容 SHA-256；
- feature schema、normalization statistics、dataset summary 与 dependency lock；
- 可移植逻辑路径，不把机器绝对路径写入身份。

checkpoint、run metadata、test metrics 与 aggregate 均携带相同的 `provenance_sha256`。Trainer 要求合法的 64 位身份；resume、evaluation 和 aggregation 对缺失或不一致 provenance fail closed。当前四个可信三种子实验共享同一 provenance。

### A05 — PARTIAL：机器路径已脱敏；被试衍生产物许可仍需确认

当前可发布审计与聚合文件使用仓库相对路径，不应包含本机用户名或绝对目录。公开 release whitelist 排除 data、checkpoint、逐步 prediction、decoded event trace、run metadata、失败运行和旧 text/windowed 结果。

剩余风险：逐步预测含时间戳、真实标签和模型输出，事件文件含真实区间，均属于人类被试衍生数据。即使不含姓名，也不能自动视为可公开数据。任何扩大共享范围的决定都必须单独核对 consent、IRB/伦理审批和数据许可。

### A06 — CLOSED / SCIENTIFIC WARN：强基线结构与活动参数公平性已补强

无文本协议下，完整多任务 loss 反传后实际获得梯度的参数量为：

| 模型 | Allocated | Executed | 相对 EPT-Net |
|---|---:|---:|---:|
| EPT-Net | 424,396 | 358,604 | 1.000× |
| Early-fusion GRU | 451,206 | 358,926 | 1.001× |
| Fusion Transformer | 415,718 | 348,046 | 0.971× |

GRU 的 hidden size 已调整，使 executed 参数与 EPT-Net 相差 +0.09%；Transformer 相差 −2.94%。`parameter_fairness.json` 保存计算协议和明细。Transformer 已加入动态因果正弦位置编码，编码与总序列长度无关，并有 dtype/device 与 truncated-prefix 等价测试。

剩余科学限制：调参只发生在当前单 session pilot 上，不能宣称穷尽所有强基线或公平覆盖同等搜索预算；论文必须披露搜索空间、预算和选择依据。

### A07 — CLOSED：重叠训练窗的重复样本权重已消除

可信训练不再从重叠窗口产生梯度，而是对每个 session 的唯一连续行每 epoch 各计一次。类别与边界权重也从同一唯一训练行集合解析，因此损失权重估计和实际优化分布一致。旧 windowed 结果不进入正式报告或图表。

### A08 — CLOSED：配置采用严格 schema

`config.py` 显式定义顶层与各 section 的允许/必需字段；未知 key、缺失 key、错误类型、非有限值、非法概率/阈值、维度/head 不整除、无效 cache/read width、错误 sequence protocol 与非 1 batch size 均被拒绝。测试覆盖未知 key、缺失 key、范围错误及跨字段约束。resolved config 与 checkpoint schema version 一并留档。

### A09 — OPEN / MEDIUM：full-state checkpoint 仍依赖不受限 pickle

manifest 指向的数据 tensor 使用 restricted loader；但训练恢复需要 optimizer、scheduler、RNG 与 NumPy 状态，当前 full-state checkpoint 仍通过 `torch.load(..., weights_only=False)` 加载。反序列化不可信 `.pt` 可能执行任意代码，而 checkpoint 内嵌 provenance 无法在反序列化之前提供安全边界。

控制：只允许加载本项目本机生成、位于受控 run 目录的 checkpoint；`.pt` 不进入公开白名单。后续如需分发权重，应另存 safetensors 或受限 model-only state，并提供外部 SHA-256。

### A10 — OPEN / MEDIUM：在线 alert 与 finalized interval 必须继续分开表述

decoder 在首次过阈值时保存 `emit_step`，而完整事件对象在后续负步或流结束时 close。现有 TTD 因此是首次帧级在线 alert latency，不是最终区间可用时间。

当前报告已经明确该语义，避免把两者混用。若论文主张实时区间输出，还需让接口显式返回 open/update/close 状态，并单独评估 close latency。

### A11 — CLOSED：resume 具备逐状态 bitwise 等价证据

checkpoint schema v3 保存 model、optimizer、scheduler、AMP scaler、全局 RNG、CUDA RNG、DataLoader generator 与 history。resume 拒绝 output directory、sequence protocol、patience、关键训练配置或 provenance 变化，并要求原 run 的 `best.pt` 存在。

CPU 集成测试将 4 epochs 连续训练与 2 epochs + resume 至 4 epochs 比较，逐项验证 model、optimizer、scheduler、scaler、RNG 与 history 完全相同。测试包含 dropout、shuffle 与实际 scheduler 状态变化，因而不是空洞的等价检查。

### A12 — PARTIAL：公开复现入口已补齐主体，治理文件仍缺

已提供 `pyproject.toml`、开发/作图依赖声明、依赖 lock、`REPRODUCIBILITY.md`、pytest 配置，以及 PowerShell/Bash 正式矩阵入口。公共 clone 缺少私有数据时，真实数据 integration test 明确 skip；有授权数据时仍严格检查哈希、split 与 tensor contract。冒烟测试脚本和冒烟测试用例已撤除，当前门禁依赖完整单元/集成套件。

未关闭项：仓库尚无明确 LICENSE、CITATION 和托管 CI；CUDA/PyTorch 安装仍需按目标 driver 单独选择。因此可以发布审计版代码，但不能称为治理材料完全的 archival release。

### A13 — CLOSED：正式实验矩阵已实现失败即停的端到端编排

`run_formal_experiments.ps1` 与对应 shell 入口显式枚举 config × seed，串行使用单 GPU；每个 seed 先训练、再评估，并验证 checkpoint 与 metrics 存在；三 seed 完成后才聚合。可信数值矩阵包含 EPT-Net、GRU、Transformer 和 `no_persistent` 配置；text-enabled 与旧 windowed 实验不在矩阵中。报告生成器验证预期 seeds、设备、determinism、156 步/30 事件、provenance 和参数公平性，缺项时拒绝输出局部均值。数值产物通过门禁不等于消融名称具备构造有效性，机制解释仍受 A15 约束。

三个 seed-42 诊断只作探索性证据，不得与三种子主结果混为显著性结论，也不得在存在联合干预时作单机制归因。

### A14 — CLOSED：运行中 manifest/tensor 变化会立即失败

首次 pilot 暴露过一次真实问题：训练过程中 prepared manifest 被另一流程替换，旧路径在后续 epoch 消失，导致运行中数据语义不稳定。该失败运行已移动到隔离目录且不发布。

修复后，`StitchedManifestDataset.materialize()` 在优化开始前一次性加载完整 session；训练和评估都在 materialization 前后重算完整 provenance，任一源码、manifest 或 tensor 内容变化都会报错退出。该设计同时防止“前几轮读旧文件、后几轮读新文件”的隐蔽混合运行。

### A15 — OPEN / HIGH：两个消融名称与实际干预不等价

`ablation_no_persistent.yaml` 设置 `persistent_state=false`，只使 `PersistentMultimodalUpdate` 绕过 attention/GRU recurrent fusion 并使用 `current_only`。主循环下一步仍把上一步 state 传入 `EventGuidedReader`；reader 的 adaptive policy 明确拼接 `previous_state`。因此该配置仍具有跨步条件信息，准确名称应为 **no recurrent fusion update**，不能称为完全 memoryless，也不能用其结果单独证明 persistent memory 的作用。

`ablation_single_scale.yaml` 同时设置 `use_eeg_spec=false` 和 `use_hr=false`。在当前可信输入中，它实际把模型改成 EEG-time + behavior only，同时改变 temporal representation 与模态集合；这不是只改变一个 scale 的干净干预。

影响：两项结果都可以作为对应联合配置的描述性数值，但不得用于“persistent state 有/无效”或“single-scale 优/劣”的因果结论。

关闭条件：新增真正 memoryless 的 reader/update 路径，确保 reader policy 不读取任何先前 state；另建保持 active modality 集合不变、只改变 EEG temporal-scale construction 的对照。为两项加入结构契约测试，并按预注册三种子协议重跑。

## 实验结果对代码主张的约束

连续会话三种子结果显示：

- EPT-Net event mAP：0.0288±0.0316；
- Early-fusion GRU event mAP：0.0844±0.0208；
- Fusion Transformer event mAP：0.0865±0.0667；
- Config `no_persistent`（实际仅 no recurrent fusion update）event mAP：0.0449±0.0019。

因此，当前实现不能支持“完整模型优于强基线”。由于 `no_persistent` 并非 memoryless，其数值也不能用于判断 persistent memory 是否带来稳定收益；同理，当前 `single_scale` 不能隔离 scale 效应。这些不是需要从审计中隐藏的问题，而是必须写入论文限制并由新消融修正的构造有效性缺口。

## 验证命令与结果

从 `code/` 执行：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest -q
python -m compileall -q src tests
python -m ruff check src tests
```

本轮复核：

- pytest：`157 passed, 19 warnings`；warning 均来自 PyTorch Transformer 在 `norm_first=True` 下未启用 nested-tensor fast path 的性能提示，不影响正确性；
- compileall：PASS；
- Ruff：PASS；
- machine-readable audit JSON：可解析；
- 审计文件路径/用户名扫描：无机器特定绝对路径或本机用户名。

公开 staging 目录仍须按 `docs/RELEASE_MANIFEST.md` 执行机器路径、用户名、凭据模式和超大文件扫描；扫描规则保留在发布流程中，不把任何机器特定示例写入审计产物。

## 发布闸门

- [x] 训练、验证、测试统一为完整连续 session，撤销重叠窗优化。
- [x] 正式主矩阵四个实验均完成三种子训练、评估与 aggregate。
- [x] 秒级 alert TTD、因果位置编码与 prefix invariance 有回归测试。
- [x] provenance 绑定源码、manifest、有序 tensor 和必需 prepared artifacts；resume/eval/aggregate fail closed。
- [x] 严格配置 schema、checkpoint schema v3、bitwise resume equivalence 与防覆盖保护完成。
- [x] 基线按 executed 参数审计；text-enabled 与旧 windowed 结果隔离。
- [x] 审计文件移除机器绝对路径与本机用户名。
- [ ] 获得多 participant/session 数据并完成 subject/session-disjoint 外部测试。
- [ ] 补齐上游 feature extractor 的代码、权重、拟合范围与因果 provenance。
- [ ] 解释并在新数据上复验完整模型弱于基线的结果。
- [ ] 新增真正 memoryless 的状态消融，以及模态集合不变的纯 temporal-scale 消融；三种子重跑后再讨论机制贡献。
- [ ] 确认所有被试衍生产物的公开许可；公开包坚持 whitelist。
- [ ] 增加 LICENSE、CITATION、托管 CI 与独立外部复核。

最终判断：**工程复现门禁通过；科学外部效度与模型主张门禁不通过。**
