# EPT-Net：冻结 aligned11 完整实验

EPT-Net 用因果 EEG、PPG 生理信号和视频历史持续估计目标事件概率并定位事件区间。本仓库的固定标签语义是 `0 = deception`、`1 = truth`，因此正类始终为 `0`；它不是 session 级二分类任务。

当前唯一可执行和可报告的主路线使用冻结数据集：

```text
bci_subjects_ept_v6_marlin4060_aligned11
```

它包含 11 个 session，`session_011` 被明确排除。按当前指定协议，`sessions_all.jsonl` 中的 11 个 session 全部参与训练；原 validation/test manifest 保持不变并用于固定评估视图。由于这些评估 session 已参与训练，输出属于训练内评估，不是 held-out 泛化结果。该路线只读取已完成的对齐和特征，不会重新运行 Whisper、MARLIN、重对齐或重标注。

## 已验证的服务器环境

以下配置已在 2026-09-30 的目标 AutoDL 实例上验证：

| 项目 | 当前值 |
|---|---|
| 项目目录 | `/root/EPT2026` |
| 数据入口 | `/root/EPT2026/data -> /root/autodl-tmp/.autodl/data` |
| 冻结数据 | `/root/data/processed/bci_subjects_ept_v6_marlin4060_aligned11` |
| GPU | NVIDIA GeForce RTX 4090，24 GB |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| 字体 | Arial 已安装并可被 Matplotlib 精确解析 |
| 验证提交 | `ea4555e69af489573081689b9d922afd6fb8ec51` |

## 1. 一次性配置环境

安装完成后，所有常用动作也可以通过统一的 `ept` 入口调用：

```bash
ept doctor --config configs/experiments/aligned11_eptnet.yaml --data-root /root/data
ept resolve-config --config configs/experiments/aligned11_eptnet.yaml \
  --output results/aligned11_eptnet/resolved.yaml
ept train --config configs/experiments/aligned11_eptnet.yaml --device cuda:0
ept evaluate --help
```

`ept train` 和 `ept evaluate` 使用新的 `eptnet.training` / `eptnet.evaluation`
域模块；旧的 `python -m eptnet.train`、`python -m eptnet.evaluate` 命令仍由
兼容门面支持。目录边界和复现记录见 [`docs/architecture.md`](docs/architecture.md)
与 [`docs/reproducibility.md`](docs/reproducibility.md)。

服务器基础环境已带 CUDA PyTorch。项目虚拟环境复用该 PyTorch，只在 `.venv` 内安装仓库锁定依赖。基础环境原有 SciPy 二进制存在运行时不一致，因此最后一条安装命令会在 `.venv` 中重装同版本 NumPy/SciPy；版本没有改变。

```bash
cd /root/EPT2026

test -x .venv/bin/python || \
  /root/miniconda3/bin/python -m venv --system-site-packages .venv

.venv/bin/python -m pip install \
  -r requirements-lock.txt \
  -r requirements-figures.txt \
  -e .

.venv/bin/python -m pip install --ignore-installed --no-deps \
  numpy==1.26.4 scipy==1.16.3

.venv/bin/python -m pip check
.venv/bin/python -c "import torch, scipy, seaborn; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(scipy.__version__, seaborn.__version__)"
```

## 2. 验证完整链路，但不跑完整实验

下面的命令是本次配置确认所使用的入口。它会执行：

- 全部单元测试；
- 七个正式配置的命令 dry-run；
- 冻结数据、all-11 训练 manifest、固定评估 manifest、session tensor SHA-256、GPU 和参数公平性预检；
- 七个配置各 1 epoch、1 个完整训练 session、1 个完整验证 session 的 CUDA smoke；
- DEMO 图片的 PDF/SVG/500 dpi PNG 导出与 QA。

它不会读取 test 指标，也不会启动完整实验：

```bash
cd /root/EPT2026
bash scripts/experiments/run_marlin11_complete.sh validate
```

Smoke 产物位于 `results/validation/<UTC时间>/`。DEMO 图片带有明确的 `DEMO` 标识，不得作为实验结果。

## 3. 一口气跑完全部正式实验、评估和图片

环境和 smoke 验证通过后，只需这一条命令：

```bash
cd /root/EPT2026
bash scripts/experiments/run_marlin11_complete.sh formal
```

该入口按顺序完成：

1. 再次执行测试、命令检查和冻结数据预检；
2. EPT-Net 主模型使用 seed `13`；Early-fusion GRU 与 Fusion Transformer 使用 seed `42`；
3. Video-only 输入诊断使用 seed `42`；
4. Fixed-reader 与 No-persistent-state 机制诊断，seed `42`；
5. 每个 run 的最佳 checkpoint 在固定训练内评估视图上计算指标、逐步预测、事件解码和跨 seed 聚合；
6. 从冻结 test manifest 的首个 session 预声明定性样本；
7. 生成正式动态跟踪图，以及 Full/Video 两组探索性对比图，包括 PDF、SVG、500 dpi PNG 与 QA 报告。对比图允许 seed 不同，但必须显示真实 seed。

定性样本会在训练前写入 `results/marlin11_shortpaper_figure_sample_id.txt`。重复运行必须与该记录一致，避免根据图形效果事后挑选样本。

### 后台运行

完整实验建议在后台运行：

```bash
cd /root/EPT2026
mkdir -p logs
nohup bash scripts/experiments/run_marlin11_complete.sh formal \
  > logs/marlin11_complete.log 2>&1 &
echo $! | tee logs/marlin11_complete.pid
```

监控：

```bash
tail -f /root/EPT2026/logs/marlin11_complete.log
ps -fp "$(cat /root/EPT2026/logs/marlin11_complete.pid)"
nvidia-smi
```

若任务被正常中断，重新执行同一条 `formal` 命令即可。入口会严格验证并复用已完成 run，只对同时存在可信 `last.pt` 和 `best.pt` 且尚无最终指标的当前短论文 run 执行续跑。它不会删除或静默覆盖旧结果。

## 4. 实验矩阵

| 分组 | 配置 | Seeds |
|---|---|---|
| 主比较 | EPT-Net EEG+PPG+Video | 13 |
| 主比较 | Early-fusion GRU EEG+PPG+Video | 42 |
| 主比较 | Fusion Transformer EEG+PPG+Video | 42 |
| 输入诊断 | EPT-Net Video only | 42 |
| 机制 | EPT-Net Fixed reader | 42 |
| 机制 | EPT-Net No persistent state | 42 |

所有短论文配置关闭 audio 和 text。训练读取冻结特征，不重新调用大型行为编码器。

每个正式 epoch 的训练 cohort 都是同一组 11 个 session。validation/test 仍沿用冻结 manifest 以保持指标和图片流水线不变，但不具备训练外独立性。

## 5. 输出位置

每个正式 run 写入独立目录：

```text
results/<experiment>/seed_<seed>/
├── resolved_config.yaml
├── run_metadata.json
├── history.json
├── best.pt
├── last.pt
├── test_metrics.json
├── test_predictions.jsonl
├── test_events.json
└── figures/
    ├── training_history.csv
    ├── fig_training_dynamics.pdf
    ├── fig_training_dynamics.png
    └── training_figure_manifest.json
```

各实验目录还包含 `aggregate.json` 和 `aggregate.csv`。正式论文图位于：

```text
fig/fig04_dynamic_tracking/
├── fig04_dynamic_tracking.pdf
├── fig04_dynamic_tracking.svg
├── fig04_dynamic_tracking.png
└── fig04_dynamic_tracking.qa-report.json

fig/fig05_modality_evidence/
├── fig05_modality_evidence.pdf
├── fig05_modality_evidence.svg
├── fig05_modality_evidence.png
└── fig05_modality_evidence.qa-report.json

```

## 6. 安全与证据边界

- 不得重新生成、修改或修补冻结数据、manifest、划分、标签或对齐结果。
- 不得把 dry-run、preflight、smoke 或 DEMO 图片报告为科学实验结果。
- 不得把当前 validation/test 指标描述为 held-out、跨受试者泛化或独立测试结果。
- 不得删除或覆盖已有正式 result 目录来绕过失败。
- 只有 `test_metrics.json`、逐步预测、事件结果和通过身份校验的 aggregate 才能进入正式结果与图片。
- `0 = deception`、`1 = truth` 和正类 `0` 是全仓库固定语义。
- 模型随机性与跨 participant 不确定性必须分开报告。

## 7. 手动入口与详细协议

统一入口内部调用：

```text
scripts/experiments/run_marlin11_shortpaper.sh
scripts/experiments/preflight_marlin11_shortpaper.py
scripts/experiments/verify_marlin11_shortpaper_results.py
fig/generate_all.py
```

需要拆分运行、审查恢复条件或查看图形数据契约时，参阅：

- [`README_SERVER.md`](README_SERVER.md)：服务器逐步操作和故障策略；
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)：冻结证据与复现协议；
- [`fig/FIGURE_CONTRACTS.md`](fig/FIGURE_CONTRACTS.md)：正式图片输入、样本选择和 QA 契约；
- [`docs/experiment_design.md`](docs/experiment_design.md)：研究问题与实验设计。
