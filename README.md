# EPT-Net

Executable research code for **Event-guided Persistent Temporal Network (EPT-Net)**. The current priority route is the 12-session cohort in which video, audio, and annotated text are all defensibly paired. A separate 29-session registered-clock no-behavior route is retained as a physiological baseline; their datasets, checkpoints, and reported results must not be mixed.

## Implemented system

- native-rate EEG/HR-window and word-aligned precomputed-feature input paths;
- branch-specific bounded physiological caches and differentiable interval reading;
- local evidence encoders with an optional latest-state bypass;
- persistent multimodal attention and confidence-controlled gated state update;
- class, start/end-boundary, and boundary-offset heads with masked multi-task loss;
- incremental, prefix-only interval decoding and frame/boundary/event/latency metrics;
- causal GRU and Transformer baselines plus modality/mechanism ablations;
- deterministic source staging, SHA-256 provenance, train-only preprocessing, manifests, and compact window serialization;
- de-duplicated complete-session timelines for training, validation, and test;
- participant-balanced checkpoint selection and validation-only threshold calibration;
- subject-macro test metrics, participant bootstrap intervals, pooled diagnostics, and a fixed-threshold sensitivity analysis.
- threshold-free event AP/mAP from one offset proposal per target-valid step, kept separate from thresholded online event F1 and latency.

## 当前优先入口：12-session 全模态完整训练

`configs/eptnet_v6_marlin12_seed42.yaml` 是当前优先训练配置。候选 cohort 只包含以下 12 个同时具有已配对视频、音频和标注文本的 session：

本节是 RTX 4090/通用高显存入口。RTX 4060 使用独立的 [`README_4060.md`](README_4060.md)、配置、缓存、processed dataset 和结果目录；运行 4060 入口不会写入本节的 4090 路径。

```text
session_002, session_003, session_004, session_006,
session_008, session_009, session_010, session_011,
session_012, session_015, session_017, session_018
```

数据准备在任何归一化和 split 之前排除其余 session，再以固定 seed 生成 subject-disjoint train/validation/test 划分。完整训练显式开启全部六条输入分支：

```text
train (8): session_002, session_003, session_004, session_006,
           session_009, session_010, session_012, session_018
val   (1): session_011
test  (3): session_008, session_015, session_017
```

该划分已经用相同标签、时间轴、模态可用性约束做过预检；assignment SHA-256 为 `379545df499103693d277be103bfca6cee39d95bffe9bf30df8f262ff59b873c`。

| 分支 | 状态 | 输入/编码器 |
|---|---:|---|
| EEG 时域 | 开启 | 8 维预计算特征 → MLP 投影 |
| EEG 频域 | 开启 | 40 维五频带特征 → MLP 投影 |
| PPG 生理 | 开启 | 双通道 40 维统计特征 → MLP 投影 |
| 视频 | 开启 | MARLIN ViT-Small YTF，384 维 |
| 音频 | 开启 | Microsoft WavLM Base+，768 维 |
| 文本 | 开启 | Chinese MacBERT-base，768 维 |

三种行为编码器在离线准备阶段冻结。MARLIN、WavLM 和 MacBERT 都使用 `cuda:0`；训练阶段读取已缓存特征，不重复运行大模型。MARLIN 对每个一秒时间步采样 16 帧，并对每个采样帧独立运行 FaceX-Zoo 检测与裁剪，不复用一秒内的裁剪框；该策略带有独立版本号并进入缓存键。六个分支都在配置中开启，但原始 EEG/PPG 在个别 session 缺失时仍由 `physiology_mask` 正确屏蔽，绝不伪造观测。

### 1. 进入项目、安装依赖并确认 GPU

以下命令使用 PowerShell。无论终端当前位于哪里，都先进入 `code/`：

```powershell
Set-Location F:\EPT-Net\code

.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-behavior-lock.txt
.\.venv\Scripts\python.exe scripts\data\install_behavior_encoder_assets.py

.\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print(torch.__version__, torch.version.cuda); print(round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GiB')"
```

4090 服务器应显示 `NVIDIA GeForce RTX 4090` 和约 `24 GiB`。模型文件与原始数据门禁：

```powershell
if (-not (Test-Path .\src\eptnet\models\behavior\face\marlin\assets\marlin_vit_small_ytf.encoder.pt)) { throw "MARLIN checkpoint missing" }
if (-not (Test-Path .\data\raw\bci_subjects_ept_v1)) { throw "12-session raw root missing" }
```

在耗时编码前，确认筛选器得到的行为全模态 cohort 恰好是预注册的 12 个 session：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe -c "from pathlib import Path; from eptnet.data.prepare_bci_subjects import discover_sessions; xs=discover_sessions(Path('data/raw/bci_subjects_ept_v1')); got=[x.session_id for x in xs if x.video_file is not None and x.audio_file is not None and x.annotation is not None]; expected=['session_002','session_003','session_004','session_006','session_008','session_009','session_010','session_011','session_012','session_015','session_017','session_018']; assert got==expected,(got,expected); print('12-session source gate: PASS')"
```

### 2. 生成 12-session 全模态数据

先生成强制的真实时间对齐。Whisper `turbo` 在外部音频上产生词级时间戳，标注稿只通过精确字符匹配映射到这些时间戳；视频内嵌音轨再与外部音频做互相关，得到视频相对音频的偏移。任一 session 的精确字符覆盖率低于 50% 或音轨相关性不达标都会停止；未匹配字符保持无时间戳，不会回退到按全文长度均匀拉伸：

```powershell
.\scripts\data\prepare_whisper_alignment12.ps1
```

该步骤可原样重跑，已有且源文件 SHA-256、Whisper 模型均一致的 session 会显示 `alignment_cache_hit`。完成后再用 RTX 4090（24 GB）生成冻结特征：

```powershell
.\scripts\data\prepare_current_session_registered_marlin12.ps1 `
  -BehaviorDevice cuda:0 `
  -BehaviorBatchSize 16
```

逐帧独立检测的质量优先策略计算量较大：本地 RTX 4060 预计约 5.5–7 小时；RTX 4090 预计约 2–4 小时。实际时间还取决于服务器 CPU 视频解码和磁盘吞吐。`BehaviorBatchSize=16` 只加速 MARLIN/WavLM/MacBERT 批推理，不会把串行的人脸检测缩短 16 倍。

MARLIN 的完成结果缓存到 `data/cache/bci_subjects_ept_v6_marlin_complete12/marlin/`。若命令中断，重新执行时加 `-Overwrite`；它只重建未完成的 processed 目录，已经写好的 MARLIN cache 会显示 `MARLIN cache hit`，无需重新提取：

```powershell
.\scripts\data\prepare_current_session_registered_marlin12.ps1 `
  -BehaviorDevice cuda:0 `
  -BehaviorBatchSize 16 `
  -Overwrite
```

MARLIN 不再把整段视频均匀压缩到文本长度，而是按每个音频秒对应的真实视频时间采样 16 帧；仍对 16 帧逐帧运行 FaceX-Zoo。第一次启用人脸裁剪时，官方 MARLIN 依赖会在本地 asset 目录安装 FaceX-Zoo。

在另一个 PowerShell 窗口查看实时进度（停止此监视器不会停止数据生成）：

```powershell
Set-Location F:\EPT-Net\code
.\scripts\data\watch_marlin12_progress.ps1
```

```powershell
Test-Path .\data\processed\bci_subjects_ept_v6_marlin_complete12\_SUCCESS.json
```

只有上一步返回 `True` 才能训练。随后执行完整数据契约门禁：

```powershell
$summary = Get-Content .\data\processed\bci_subjects_ept_v6_marlin_complete12\dataset_summary.json -Raw | ConvertFrom-Json
if ($summary.num_subjects -ne 12) { throw "Expected 12 sessions, got $($summary.num_subjects)" }
if ($summary.split_sessions.train.Count -ne 8 -or $summary.split_sessions.val.Count -ne 1 -or $summary.split_sessions.test.Count -ne 3) { throw "Unexpected 8/1/3 split" }
if ($summary.behavior_features.video_backend -ne "marlin") { throw "Video backend is not MARLIN" }
if ($summary.behavior_features.audio_backend -ne "wavlm") { throw "Audio backend is not WavLM" }
if ($summary.behavior_features.text_backend -ne "macbert") { throw "Text backend is not MacBERT" }
$summary.split_sessions | Format-List
```

再确认训练配置没有关闭任何模态：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe -c "from eptnet.config import load_config; c=load_config('configs/eptnet_v6_marlin12_seed42.yaml'); assert (c['data']['video_dim'],c['data']['audio_dim'],c['data']['text_dim'])==(384,768,768); keys=('use_video','use_audio','use_text','use_eeg_time','use_eeg_spec','use_hr'); assert all(c['model'][k] for k in keys); print('12-session MARLIN config gate: PASS')"
```

### 3. Smoke 验证

Smoke 使用一个完整训练 session 和一个完整验证 session，仅验证端到端 CUDA、损失、checkpoint 和进度条，不读取 test 指标：

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_v6_marlin12_seed42.yaml `
  -Seed 42 `
  -Device cuda:0 `
  -Smoke
```

### 4. 完整训练

Smoke 成功后运行正式 12-session 训练。不要添加 `-Smoke`：

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_v6_marlin12_seed42.yaml `
  -Seed 42 `
  -Device cuda:0
```

恢复中断训练：

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_v6_marlin12_seed42.yaml `
  -Seed 42 `
  -Device cuda:0 `
  -Resume results\eptnet_v6_marlin12_gated_seed42\seed_42\last.pt
```

最终评估只读取训练选出的 `best.pt`，阈值只在 validation 上校准，然后冻结并评估 test：

```powershell
.\scripts\experiments\evaluate.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_v6_marlin12_seed42.yaml `
  -Checkpoint results\eptnet_v6_marlin12_gated_seed42\seed_42\best.pt `
  -Device cuda:0
```

正式产物分别位于：

```text
data/processed/bci_subjects_ept_v6_marlin_complete12/
data/cache/bci_subjects_ept_v6_marlin_complete12/marlin/
results/eptnet_v6_marlin12_gated_seed42/seed_42/
```

## 29-session 生理模态对照入口

`configs/eptnet_v6_seed42.yaml` 是当前数据的单 seed 完整训练配置。它固定 `seed=42`，读取 20 个完整训练 session 和 4 个完整验证 session，每个 session 每 epoch 恰好产生一次 optimizer update；训练完成后再由独立评估入口读取 5 个 test session。当前 v6 数据明确排除了 20 个 legacy facial CSV，且没有 cohort-wide 可信的 OpenFace/WavLM/MacBERT 缓存，因此正式配置关闭 video/audio/text，不能把零占位或单样本 extractor smoke 冒充多模态结果。

### 网络与优化目标

完整 EPT-Net 保留论文构想中的 read–update 闭环：三个生理分支先投影到共享隐空间并写入各自的因果缓存；branch-specific reader 根据上一状态、当前可用行为上下文和最新生理状态预测读取中心与跨度；可微区间采样和局部 Transformer 形成证据 token；多头注意力完成跨分支融合。persistent update 使用可审计的写入门与保留门：

```text
candidate_t = GRU(fused_evidence_t, state_{t-1})
state_t = write_t * candidate_t
        + (1 - write_t) * retention_t * state_{t-1}
```

没有任何可用 token 时严格保持旧状态。latest bypass 无 bias，缺失当前值不能产生伪常数证据。输出同时包含逐步分类、开始/结束边界与左右 offset；训练目标为 masked class CE、boundary BCE、positive-step SmoothL1 和不跨真实标签边界的平滑正则。`write_gates`、`retention_gates`、reader center/width/weight 都保留在模型输出中，供后续机制分析。

### 1. 环境与数据门禁

以下命令均从 `code/` 执行。先安装与本机 CUDA/驱动匹配的 PyTorch，再安装锁定依赖。`matplotlib` 属于正式训练依赖，因为每次训练必须自动生成图，而不是训练后手工补图。

```powershell
cd path\to\EPT-Net\code
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path

# 应输出 True、NVIDIA GeForce RTX 4060 Laptop GPU（或实际指定设备）
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 数据必须已经存在且完成冻结写入
Test-Path .\data\processed\bci_subjects_ept_v6_session_registered_no_facial\_SUCCESS.json
```

如需从当前规范化 raw 重新生成 v6，使用：

```powershell
.\scripts\data\prepare_current_session_registered_no_facial_sessions.ps1
```

### 2. 真实 session smoke 与耗时估算

该 smoke 只运行 1 个 epoch，使用一个完整真实训练 session 和一个完整真实验证 session，不截断序列、不访问 test、不做阈值选择，也不构成实验结果。它验证 CUDA 前后向、自动权重、进度条、best/last checkpoint、断点状态、训练图和 runtime estimate 的完整链路。

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_v6_seed42.yaml `
  -Seed 42 `
  -Device cuda:0 `
  -Smoke
```

默认 smoke 目录为：

```text
results/smoke/eptnet_v6_gated_seed42_no_text_smoke/seed_42/
```

本机 RTX 4060 Laptop GPU 的 2026-09-28 最终 smoke 测量为：完整 404-step train session 约 7.82 s，完整 402-step validation session 约 2.29 s。按完整协议每 epoch 的 8,222 train steps 与 1,693 validation steps 外推，点估计约 168.9 s/epoch；100 epoch 约 4.69 h，保守 ±35% 区间约 3.05–6.33 h。若在 patience 首次允许的位置停止，16 epoch 点估计约 0.75 h。该估计不含最终 test evaluation 与任何离线特征提取；以当前 run 的 `runtime_estimate.json` 为准。

### 3. 启动 seed 42 完整训练

下面的命令会在前台运行并显示 epoch、train batch 和 validation batch 三层进度信息。tqdm 自带 elapsed/ETA，postfix 显示 rolling loss、learning rate、best validation loss 和 stale/patience。不要添加 `-Smoke`。

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_v6_seed42.yaml `
  -Seed 42 `
  -Device cuda:0
```

输出目录：

```text
results/eptnet_v6_gated_seed42_no_text/seed_42/
├── resolved_config.yaml
├── run_metadata.json
├── history.json
├── best.pt
├── last.pt
└── figures/
    ├── training_history.csv
    ├── fig_training_dynamics.pdf
    ├── fig_training_dynamics.png
    └── training_figure_manifest.json
```

PDF 是论文排版用矢量图，PNG 为 300 DPI；使用 Okabe–Ito 色盲友好配色、颜色以外的 marker/line-style 冗余编码和 Times-compatible serif 字体。训练图只表示优化过程，不包含 test 指标。

### 4. 中断恢复、重绘与最终评估

恢复必须使用同一配置、seed、output directory 与未变化的代码/数据 provenance：

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_v6_seed42.yaml `
  -Seed 42 `
  -Device cuda:0 `
  -Resume results\eptnet_v6_gated_seed42_no_text\seed_42\last.pt
```

从已保存 `history.json` 独立重绘训练图：

```powershell
.\.venv\Scripts\python.exe scripts\reporting\gen_fig_training.py `
  --run-dir results\eptnet_v6_gated_seed42_no_text\seed_42
```

训练完成后，只用 `best.pt` 执行 validation-only threshold calibration 和冻结后的 held-out test：

```powershell
.\scripts\experiments\evaluate.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_v6_seed42.yaml `
  -Checkpoint results\eptnet_v6_gated_seed42_no_text\seed_42\best.pt `
  -Device cuda:0
```

评估默认写出 `test_metrics.json`、`test_metrics_predictions.jsonl` 和 `test_metrics_events.json`。单个 seed 42 可用于完整链路和预注册主运行，但正式论文比较仍应使用相同协议的多个 seed，并把优化随机性与跨 session/participant 不确定性分开报告。

## 12-session 行为模态实现与审计细节

面部、语音和文本先离线编码到同一 session 时间轴，再由完整 EPT-Net 与生理分支联合训练。只有编码产物、融合配置和训练门禁同时通过后，才进入正式训练；模型下载成功、单样本前向成功或占位零张量都不等同于完成多模态数据准备。

当前正式 12-session 入口是本 README 顶部的 MARLIN 路线。下文出现的 OpenFace 内容仅说明保留的对照实现，不应替换 `eptnet_v6_marlin12_seed42.yaml`、MARLIN processed dataset 或 MARLIN 结果目录。

### 当前状态

| 环节 | 状态 | 说明 |
|---|---|---|
| MARLIN ViT-Small YTF 本地权重 | 已完成 | 官方 0.3.4 checkpoint 已下载并校验 SHA-256，384 维输出 |
| OpenFace 2.2.0 本地运行包与 CEN 权重 | 已完成 | 官方运行包和四个 patch model 已下载并校验 SHA-256 |
| WavLM Base+ 本地权重 | 已完成 | 固定 revision，默认离线加载 |
| Chinese MacBERT-base 本地权重 | 已完成 | 固定 revision，默认离线加载 |
| MARLIN CUDA 前向验证 | 已完成 | RTX 4060 Laptop GPU 上得到 `[1, 384]`，模型与输入均在 `cuda:0` |
| 29-session behavior 离线编码缓存 | 不可用 | 只有 12 个 session 同时具有可靠配对的视频和音频源 |
| 12-session 三行为模态编码入口 | 已完成 | 仅保留完整视频、音频、标注文本源，使用 MARLIN/WavLM/MacBERT |
| 12-session 全模态融合配置 | 已完成 | `eptnet_v6_marlin12_seed42.yaml` 显式启用 video/audio/text，并保留生理分支 |
| 训练进度条 | 已完成 | 默认显示 epoch、train batch 和 validation batch；可用 `--no-progress`/`-NoProgress` 关闭 |

29-session no-facial 配置仍是独立基线。需要三种行为模态时，必须先运行下面的 complete12 离线编码入口，再使用专用配置训练；不能只修改旧 YAML 的三个布尔开关。

### 1. 准备环境与权重

以下命令从 `code/` 目录执行。先安装与本机 CUDA 匹配的 PyTorch，再安装锁定依赖；不要让行为模型静默使用全局环境中未经锁定的 NumPy、Transformers 或 tokenizer。

```powershell
cd path\to\EPT-Net\code

python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 按本机 CUDA/driver 安装匹配的 PyTorch 后，再安装项目依赖。
python -m pip install -r requirements-lock.txt
python -m pip install -r requirements-behavior-lock.txt

python scripts\data\install_behavior_encoder_assets.py
```

安装器是幂等的：文件完整时只核验，不重复下载；缺失或损坏时才补齐。模型位于：

```text
src/eptnet/models/behavior/
├── face/marlin/assets/                 # MARLIN-small YTF encoder checkpoint
├── face/openface/assets/               # OpenFace runtime + CEN patch models
├── audio/wavlm_base_plus/assets/       # WavLM Base+ config + weights
└── text/macbert_base/assets/           # MacBERT config/tokenizer + weights
```

这些大文件被 Git 忽略，但保留在本机并可直接使用。MARLIN 使用 CC BY-NC 4.0，OpenFace runtime 也受学术/非营利非商业研究许可证约束；二者都不得作为不受限制的商业组件重新分发。

### 2. 验证编码器

使用一段包含清晰人脸的短视频做安装级验证；该命令不训练模型，也不访问 validation/test 指标。

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe scripts\audit\verify_behavior_encoder_baselines.py `
  --device cuda:0 `
  --openface-video path\to\short_face_clip.mp4 `
  --output results\behavior_encoder_baseline_smoke.json
```

通过标准为：

- OpenFace 输出 `[T, 215]`，由 AU intensity/presence、pose 和 gaze 的一秒内统计组成；
- WavLM Base+ 输出 `[T, 768]`，每个一秒音频块独立编码，不读取未来块；
- MacBERT-base 输出 `[T, 768]`，文本输入只能来自当前已观测 bin 或 causal prefix；
- 无效时间步必须为精确零，且对应 mask 为 `false`；
- 三种特征进入同一 causal TCN 后维度正确、数值有限；
- 输出 JSON 中必须保留 `full_experiment_started=false`。

### 3. 审计 behavior 数据来源

正式编码前，应对实际 source root 运行严格数据完整性审计。不要把 20 个 legacy facial CSV 当成 OpenFace 输出；面部 baseline 必须从真实视频生成新的官方 OpenFace CSV/feature cache。

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe scripts\audit\audit_bci_session_compliance.py `
  --source path\to\behavior_source_root `
  --profile strong_behavior_sources `
  --strict `
  --json-output results\behavior_source_compliance.json `
  --markdown-output results\behavior_source_compliance.md
```

`--strict` 返回非零时必须停止。缺少视频、音频、文本时间信息或 session 映射的样本，应保持该模态 `mask=false`；不能用零占位、复制文件或 duration stretching 把缺失伪装成观测值。

### 4. 离线编码契约

下一阶段的 cohort encoder 必须逐 session 写入版本化 processed 目录，而不是训练时临时调用三个大模型。每个 session 的行为张量应满足：

```text
video          [T, 384]   # MARLIN ViT-Small YTF
audio          [T, 768]   # WavLM Base+
text           [T, 768]   # Chinese MacBERT-base
modality_mask  [T, 3]     # [video, audio, text]
timestamps     [T, 2]     # 与当前一秒 session timeline 一致
target_mask    [T]
labels         [T]
```

编码阶段必须遵守以下边界：

- 音频时间轴为行为模态的相对时间基准；Whisper 对齐文本，视频内嵌音轨互相关得到偏移，MARLIN 按真实秒取帧，不再均匀拉伸视频；
- MARLIN、WavLM 和 MacBERT 分别离线生成，MARLIN 以源视频 SHA、时间步数和裁剪策略命名缓存；单个 encoder 失败不得污染另外两个模态；
- 每个 feature cache 保存 model revision、权重 SHA-256、源文件 SHA-256、窗口/hop、pooling、时间轴版本和生成时间；
- 原始数据只读，编码结果写入新的版本目录，不覆盖 v4/v5/v6；
- split 在编码前冻结，任何 train-only normalization 或 projector 不得查看 validation/test；
- 运行结束后逐 session 核对 `T`、有限值、mask、全零缺失步和 source ledger。

当前入口只处理同时拥有可靠配对视频、音频和标注文本的 12 个 session，并在划分、归一化之前排除其余 session：

```powershell
.\scripts\data\prepare_whisper_alignment12.ps1
.\scripts\data\prepare_current_session_registered_marlin12.ps1 `
  -BehaviorDevice cuda:0
```

该命令在新的 `data/processed/bci_subjects_ept_v6_marlin_complete12/` 下写入数据，不覆盖 OpenFace 数据或现有 v6 no-facial 数据。MARLIN、WavLM 和 MacBERT 使用指定的 `cuda:0`；视频解码仍由 CPU 完成，这是 OpenCV I/O，不代表 MARLIN 主干回退到 CPU。

### 5. behavior 融合协议

complete12 主入口训练行为模态与现有 EEG/PPG 生理分支的全模态 EPT-Net。三个上游行为 encoder 均冻结并离线缓存，训练时分别投影到相同的 `model.hidden_dim`，加入 modality embedding，再使用带 `modality_mask` 的模态注意力和 masked pooling 得到每个时间步的 behavior context。融合必须满足：

- `use_behavior_context=true`，并显式开启 `use_video/use_audio/use_text`；
- `use_eeg_time/use_eeg_spec/use_hr=true`，保留当前生理分支；若另做 behavior-only 消融，必须使用独立配置与输出目录；
- video/audio/text 使用同一 train/validation/test session split；
- 模态缺失通过 mask 处理，不用可学习常数替代观测；
- 下游 causal head、损失、early stopping 和 validation-only threshold calibration 在各 behavior baseline 间保持一致；
- 若做正式消融，应同时保留 MARLIN-only、WavLM-only、MacBERT-only 与三模态融合，不能只报告融合模型。

目标配置至少应声明：

```yaml
data:
  video_dim: 384
  audio_dim: 768
  text_dim: 768

model:
  use_behavior_context: true
  use_video: true
  use_audio: true
  use_text: true
  use_eeg_time: true
  use_eeg_spec: true
  use_hr: true
```

可直接使用的 12-session 全模态配置为 `configs/eptnet_v6_marlin12_seed42.yaml`。它同时开启 video/audio/text，并保留 EEG 时域、EEG 频域和 PPG 生理分支；它不是 OpenFace 路线或 29-session no-text 基线，也不能与它们混写结果。

### 6. 训练进度条

正式训练必须默认显示进度条，而不是长时间只留下一个无输出的终端。进度显示属于运行可观测性，不得参与 loss、随机数或 checkpoint 状态，因此开启与关闭进度条必须产生相同的模型结果。

最低要求如下：

- 外层显示 `epoch current/total`；
- train 与 validation 分别显示 `batch current/total`；
- postfix 至少包含 rolling total loss、当前 learning rate、best validation loss 和 stale epochs；
- 每个 epoch 结束后仍写一条结构化 JSON，并原子保存 `history.json`、`last.pt`，改进时保存 `best.pt`；
- 交互式本地运行默认开启，CI/日志重放允许 `--no-progress`；
- 断点恢复后从 checkpoint 的下一 epoch 继续，进度条总数不得重新从 1 冒充完整训练；
- PowerShell launcher 必须保证终端中的动态进度条正常刷新，同时把稳定的 epoch JSON 写入日志，不能让 `Tee-Object` 把回车刷新字符写成大量脏行。

预期终端形式示例：

```text
Epoch 03/100 | train 12/20 | loss=0.6841 | lr=3.00e-4
Epoch 03/100 | valid  4/4  | loss=0.7028 | best=0.6985 | stale=1/15
```

截至 2026-09-28，这一门禁已经实现：`Trainer.fit()` 显示外层 epoch bar，`Trainer._epoch()` 分别显示 train/validation batch bar；每个 epoch 完成后仍输出稳定 JSON 并保存 checkpoint/history。直接执行 Python 时会自动检测终端，`train.ps1` 默认显式传入 `--progress`，并用 PowerShell transcript 代替会破坏动态刷新的 `Tee-Object`。CI 或纯日志运行可使用 `--no-progress`，PowerShell launcher 对应 `-NoProgress`。进度开关不进入模型、优化器、scheduler 或 RNG 状态，回归测试要求开关前后模型状态一致。

### 7. 开始训练、恢复与评估

完成上面的 complete12 离线编码后，使用专用配置启动训练：

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_v6_marlin12_seed42.yaml `
  -Seed 42 `
  -Device cuda:0
```

需要关闭动态显示时使用 `-NoProgress`；训练记录仍写入 `history.json` 和 checkpoint：

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_v6_marlin12_seed42.yaml `
  -Seed 42 `
  -Device cuda:0 `
  -NoProgress
```

恢复中断训练时，配置、数据 provenance、seed 和 output directory 必须与 checkpoint 一致：

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_v6_marlin12_seed42.yaml `
  -Seed 42 `
  -Device cuda:0 `
  -Resume results\eptnet_v6_marlin12_gated_seed42\seed_42\last.pt
```

训练完成后，只使用保存的 `best.pt` 做验证阈值校准和测试评估：

```powershell
.\scripts\experiments\evaluate.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_v6_marlin12_seed42.yaml `
  -Checkpoint results\eptnet_v6_marlin12_gated_seed42\seed_42\best.pt `
  -Device cuda:0
```

每个 seed 至少应产生 `resolved_config.yaml`、`run_metadata.json`、`history.json`、`best.pt` 和 `last.pt`。正式比较不得只跑一个 seed，也不得根据 test 结果修改阈值、split、encoder pooling 或模态组合。

## Environment

Python 3.10+ is required. Install a PyTorch build appropriate for the local CUDA, MPS, or CPU environment, then install the locked CPU-side dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

`requirements.txt` and `requirements-lock.txt` intentionally exclude `torch`, `torchvision`, and `torchaudio`, so the accelerator-specific build remains under user control.

## Prepare the primary 18-subject cohort

From `code/`, convert the heterogeneous per-subject collection with:

```powershell
.\scripts\data\prepare_bci_subjects.ps1 -Source ..\BCI
```

The adapter creates `data/processed/bci_subjects_ept_v1/` using one session per subject, a frozen 12/2/4 train/validation/test split, training-only normalization, and explicit masks for unavailable inputs. The split is produced before model fitting by deterministic exhaustive constrained stratification over subject-level label and modality summaries. It is label- and modality-aware but model-independent; model outcomes are never used to select or revise it.

The primary configuration is `configs/bci_subjects.yaml`. The companion `configs/bci_subjects_all_sessions.yaml` includes training and validation subjects and is therefore diagnostic inference only, never held-out evidence.

### Current staged 29-session registered-clock no-facial dataset

The normalized raw tree contains 29 annotated `session_NNN` directories. The 20
legacy facial-action CSV files remain in the centralized raw provenance folder but
are explicitly excluded from the current model-ready dataset. Build the versioned,
de-identified dataset without rewriting raw evidence:

```powershell
.\scripts\data\prepare_current_session_registered_no_facial_sessions.ps1
```

The command verifies the raw file ledger, preserves canonical session IDs, defines the
29 sessions as the supervised cohort, excludes every facial CSV record before split
construction, fits normalization on training sessions only, and writes one complete
`timeline.pt` per session without redundant overlapping-window copies. Use
`configs/bci_subjects_29_session_registered_no_facial.yaml`; its manifests point directly
to complete session timelines and disable video/audio/text model branches.
`alignment_report.json` records the clock policy and evidence. EEG first uses direct
OpenBCI row-epoch intersection. For a user-confirmed same-session pair whose device
wall clock is wholly offset, preprocessing applies one constant offset derived from an
independent session-end boundary while preserving every within-EEG gap and interval;
it never stretches EEG to transcript duration. PPG uses the acquisition-start filename
plus the independently estimated 385 Hz device rate. The prepared v6 artifact has valid
EEG in 7/8 EEG-present sessions (2,377 one-second steps) and valid PPG in 12/14
PPG-present sessions (4,637 steps). The 20 legacy facial CSVs remain excluded. The
devices did not share a hardware trigger, so this registration does not support
sub-second lag claims. The accompanying
`dataset_integrity.json`,
per-session `session_manifest.json`, and `_SUCCESS.json` bind the completed artifact.
Earlier duration-normalized and facial-enabled datasets remain historical derived
artifacts and must not be relabeled as this version.

Create the self-contained collaborator/download bundle after preparation:

```powershell
python scripts\data\package_processed_dataset.py `
  data\processed\bci_subjects_ept_v6_session_registered_no_facial
```

The command restricted-loads and checksum-verifies all 29 `timeline.pt` files, rejects
links, raw-source leakage and paths outside the dataset, writes `FILES.sha256`, and
creates `data/processed/bci_subjects_ept_v6_session_registered_no_facial.zip` plus its
`.sha256` file. The extracted archive can be read with plain PyTorch using the example
in its `README.md`; the repository is not required for basic tensor loading.

### Official behavior-feature migration

The separate `bci_subjects_ept_v2_macbert` candidate replaces the deterministic text
hash with frozen HFL Chinese MacBERT-base embeddings while preserving the original
split, timestamps, labels, masks, physiology, and legacy face tensors exactly. Build
and verify it from `code/`:

```powershell
python -m pip install -r requirements-behavior-lock.txt
.\scripts\data\prepare_bci_subjects_official_text.ps1 -Source ..\BCI
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts\audit\verify_behavior_migration.py
```

Train/evaluate this candidate only with
`configs/bci_subjects_official_text.yaml`. WavLM Base+ and OpenFace adapters are
implemented but are not enabled for the cohort: genuine raw audio/video exists for
only 1/18 sessions. See `../docs/official_behavior_baseline_migration.md`.

### Audited behavior encoder baselines

The maintained face/audio/text feature encoders now live below
`src/eptnet/models/behavior/`, grouped by modality. Their pinned upstream source
snapshots, manifests, local ignored weights, and OpenFace runtime are installed with:

```powershell
python scripts/data/install_behavior_encoder_assets.py
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe scripts/audit/verify_behavior_encoder_baselines.py `
  --device cuda:0 --openface-video path\to\a_short_face_clip.mp4
```

The selected first-tier baselines are OpenFace 2.2.0 (215 interpretable pooled
AU/presence/pose/gaze features), frozen WavLM Base+ (768), and frozen Chinese
MacBERT-base (768). All three use the same two-layer causal TCN architecture for
downstream checks. The audit is deliberately bounded and does not train a model or
evaluate the test split. See `configs/behavior_encoder_baselines.yaml` and the package
README for the exact contracts and license restrictions.

### Target-speaker decision mask

Speaker identity is not provided as a separate ground-truth field. Before inspecting any highlight/bold mark, the adapter freezes the target as the unique speaker with the largest total number of non-whitespace characters in the complete transcript. This is a retrospective, label-independent conversation-length heuristic, not participant-role metadata or an online role-discovery method; the winning character share and runner-up margin are recorded and still require confirmation by the data owner. Marks are read only afterward for labels and a fail-closed consistency audit: the frozen target must own at least 95% of all marked characters. A failed audit never causes another speaker to be selected. Residual non-target marks are recorded as discarded audit counts and cannot create deception labels or events. A step is `target_mask=true` only when the frozen target contributes at least one character and strictly more characters than all non-target speakers in that bin. Empty, non-target-dominant, and tied bins remain temporal context but do not contribute to:

- class, boundary, offset, or smoothness loss;
- automatic class/boundary weighting;
- validation threshold selection;
- frame, boundary, event, early-detection, or latency metrics.

An invalid target step also closes an active decoded event. This prevents a deception interval from being joined across interviewer/background speech while preserving wall-clock indices and latency. Causality claims apply to the model's feature stream after this offline preprocessing; they do not include target-role discovery from an unfolding conversation.

### Missingness and alignment

EEG is available for 5/18 subjects, paired PPG/physiology for 13/18, facial features for 17/18, and a frozen audio representation for none. Missing branches use zero-filled placeholders plus explicit `physiology_mask`/`modality_mask` values. Encoded branch inputs are gated where unavailable, but availability patterns remain observable and are strongly confounded with subject identity; the current pilot does not establish complete-modality performance or missingness invariance. Audio and text are disabled in the trusted primary configuration.

For the current v6 candidate, transcript characters are placed by uniform interpolation
inside speaker turns. Sensor samples are retained only under the declared clock policy:
direct absolute intersection first, followed—only for a confirmed same-session EEG
pair with no direct overlap—by one documented session-end constant offset. Missing clock
evidence and invalid or flat signal still fail closed. This fixes the invalid duration
stretching used by historical v4 and the overly strict cross-device-clock assumption in
v5, but it does not create shared hardware synchronization or support sub-second
physiological-lag claims.

Every preprocessing run now writes `session_compliance.json` and
`session_compliance.md`. The default `model_contract` gate verifies safe tensors,
targets, timelines, and explicit masks for all sessions. It is deliberately weaker
than complete-modality readiness. Before a formal audio-video-text run, require
`strong_behavior_sources` for real source media and `strong_behavior_features` for
observed OpenFace, WavLM Base+, and MacBERT outputs. See
`../docs/session_data_compliance_guide.md` for the exact restoration and audit flow.

## Input contract

The precomputed path consumes:

```text
eeg_time        [B, T, 8]
eeg_spectral    [B, T, 40]  # compatibility key for audited advanced features
physiology      [B, T, 40]
video           [B, T, D_video]
audio           [B, T, 50]
text            [B, T, 768]
sequence_mask   [B, T]       # real timeline step versus batch padding
target_mask     [B, T]       # optional decision/supervision mask; required by the primary pilot
modality_mask   [B, T, 3]
physiology_mask [B, T, 3]    # optional branch availability
labels          [B, T]       # 0=deception, 1=truth
boundaries      [B, T, 2]
offsets         [B, T, 2]
positive_mask   [B, T]
```

`sequence_mask` and `target_mask` are intentionally different: an interviewer bin can be a real context step while being invalid for target-person supervision and scoring. Dataset labels remain `0=deception` and `1=truth`; evaluation explicitly maps class 0 to the event-positive class and uses the class-0 posterior.

The original single-session feature table contains 55 EEG columns. Its auditable path retains 8 `EEG_Channel_*` and 40 `EEG_Advanced_*` columns while excluding six CSP and one LDA column whose fitting population and split are unknown. The `eeg_spectral` key is retained for interface compatibility and is not a claim that every advanced feature is a spectral coefficient.

The `raw_windows` path accepts EEG `[B,T,C_eeg,S_eeg]` and HR `[B,T,C_hr,S_hr]`, allowing future feature reconstruction without changing the reader or persistent update.

## Verification

Smoke-test entry points have been removed. Use the maintained unit/regression suite:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest -q
python -m ruff check src tests scripts
python -m compileall -q -f src tests scripts
```

The suite covers configuration inheritance, tensor contracts, source hashes, split isolation, overlap de-duplication, target-mask propagation, forward/backward behavior, exact resume, positive-class polarity, one-class metrics, one-to-one event matching, padding safety, missing-modality behavior, online emission time, and causal prefix invariance.

## Repository quality gates

Repository organization is also executable policy: `tests/test_repository_layout.py` rejects launchers at the project root, executable files directly under `scripts/`, and executable source under `results/`, `figures/`, or `logs/`. All maintained commands are indexed in `scripts/README.md`; reusable logic belongs to the `eptnet` package rather than being copied into launchers.

Build release reports and publication figures from the `code/` directory:

```powershell
python scripts/reporting/build_release_report.py
python scripts/reporting/gen_fig_experiments.py --check-only
python scripts/reporting/gen_fig_experiments.py --verify-determinism
```

## Formal local-GPU matrix

Run the authoritative primary matrix on Windows:

```powershell
.\scripts\experiments\run_bci_subjects_formal.ps1 -Python python -Device cuda:0
```

The Unix counterpart is:

```bash
bash scripts/experiments/run_bci_subjects_formal.sh
```

The matrix uses seeds 13, 42, and 73 for:

- EPT-Net;
- a capacity-matched early-fusion GRU;
- a capacity-matched causal fusion Transformer;
- the principal `no_recurrent_fusion` mechanism ablation.

The ablation removes recurrent fusion-state updating but still supplies the preceding reader state; it is not a fully memoryless model. Its executed parameter count is lower than the full model, so the result is a mechanism-plus-capacity comparison and must be described as such.

The runner first creates the de-identified data audit and parameter-fairness audit. It then trains and evaluates each seed sequentially and refuses to reuse an existing formal run directory. Each seed writes `best.pt`, `last.pt`, `resolved_config.yaml`, `run_metadata.json`, `history.json`, test metrics, step predictions, and decoded events under `results/<experiment>/seed_<seed>/`.

### Participant-balanced selection and reporting

Every complete subject/session is one batch. Each training subject therefore contributes one optimizer update per epoch. Validation losses are averaged uniformly over subjects, not target steps, and early stopping selects the checkpoint with minimum participant-mean validation total loss; target-step-weighted losses are retained only as diagnostics.

Evaluation searches the configured threshold grid on validation subjects and maximizes their mean subject-level macro-F1. The chosen threshold is frozen before test. Ties prefer the value nearest 0.5 and then the lower threshold. The test artifact reports:

- subject-macro frame, boundary, and event metrics;
- a fixed-seed 10,000-resample percentile bootstrap across held-out subjects;
- pooled metrics as secondary diagnostics;
- a full fixed-threshold-0.5 sensitivity evaluation;
- frame/boundary/event/early-detection/latency, throughput, and peak-memory fields.

Event precision/recall/F1, early detection, and latency use the causal decoder at that frozen operating threshold. Event AP at each IoU, and their mean (`event_map`), are threshold-free: every target-valid step contributes one proposal scored by its positive-class probability, with boundaries given by its predicted left/right offsets and clamped to the containing contiguous target segment. No score filtering or NMS is applied. Consequently, selected-threshold and fixed-0.5 evaluations must have identical event AP/mAP even when their event F1 differs.

The participant bootstrap is computed within a model seed. With only four test subjects, its interval is descriptive and unstable. Dispersion across three model seeds measures optimization variability and must not be called a population confidence interval.

## Secondary single-session audit

The original bundle can still be prepared with:

```bash
bash scripts/data/prepare_data.sh /path/to/BCI_original_inputs_bundle
```

Its chronological `[0,709)`, `[709,859)`, and `[859,1015)` split is useful for regression checks and architecture debugging. It is not subject-disjoint. The label stream is highly fragmented, and exact train/test reuse was found in many text embeddings; the trusted configuration disables text, while `ablation_with_text.yaml` is quarantined as a shortcut audit. Single-session results remain separate from all primary-pilot aggregates and figures.

## Scientific scope

The primary matrix is a **fixed-split subject-disjoint pilot**, not nested or repeated group evaluation. Its validation set has two subjects and its test set has four. Modality availability is sparse and confounded with subject identity; audio is absent; target identity and labels are inferred from highlighting; and stream alignment is approximate. Therefore the artifacts can demonstrate a reproducible local-GPU pipeline and quantify this frozen pilot, but they cannot by themselves establish population generalization, mechanism validity, state-of-the-art performance, or top-conference readiness.

Any stronger paper claim requires a larger and better synchronized cohort, independently validated semantic event boundaries, repeated subject-group splits or nested group cross-validation, stronger causal/missingness baselines, and participant-level paired inference. The experiment report must distinguish scientific evidence from engineering/release-gate success.
