# RTX 4060：11-session MARLIN 全模态真实对齐实验

这是 RTX 4060 专用、与 RTX 4090 路线完全隔离的入口。原始候选 cohort 有 12 个 session；`session_011` 的 Whisper/标注精确字符覆盖率只有 0.975%，因此 4060 正式入口将它作为对齐质量排除项，最终使用其余 11 人。路线使用冻结模型、逐采样帧 FaceX-Zoo 检测，并把行为编码 batch 固定为 4。

```text
4060 processed: data/processed/bci_subjects_ept_v6_marlin4060_aligned11/
4060 cache:     data/cache/bci_subjects_ept_v6_marlin4060_aligned11/marlin/
4060 alignment: data/cache/bci_subjects_ept_v6_marlin4060_complete12/whisper_alignment/
4060 config:    configs/eptnet_marlin11_aligned_seed42.yaml
4060 results:   results/eptnet_marlin11_aligned_seed42/

改进训练 config:  configs/eptnet_marlin11_aligned_windowed_seed42.yaml
改进训练 results: results/eptnet_marlin11_aligned_windowed_seed42/
GRU baseline:      configs/gru_marlin11_aligned_windowed_seed42.yaml

4090 processed: data/processed/bci_subjects_ept_v6_marlin_complete12/
4090 cache:     data/cache/bci_subjects_ept_v6_marlin_complete12/marlin/
4090 results:   results/eptnet_v6_marlin12_gated_seed42/
```

## 1. 环境与 GPU 门禁

以下命令均在 PowerShell 中执行：

```powershell
Set-Location F:\EPT-Net\code

.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-behavior-lock.txt
.\.venv\Scripts\python.exe scripts\data\install_behavior_encoder_assets.py

.\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); name=torch.cuda.get_device_name(0); assert '4060' in name,name; print(name, round(torch.cuda.get_device_properties(0).total_memory/1024**3,1),'GiB')"
```

确认权重和原始的 12 人候选 cohort；后续质量门禁会排除 `session_011`：

```powershell
if (-not (Test-Path .\src\eptnet\models\behavior\face\marlin\assets\marlin_vit_small_ytf.encoder.pt)) { throw "MARLIN checkpoint missing" }
if (-not (Test-Path .\data\raw\bci_subjects_ept_v1)) { throw "raw dataset missing" }

$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe -c "from pathlib import Path; from eptnet.data.prepare_bci_subjects import discover_sessions; xs=discover_sessions(Path('data/raw/bci_subjects_ept_v1')); got=[x.session_id for x in xs if x.video_file is not None and x.audio_file is not None and x.annotation is not None]; expected=['session_002','session_003','session_004','session_006','session_008','session_009','session_010','session_011','session_012','session_015','session_017','session_018']; assert got==expected,(got,expected); print('RTX 4060 12-session source gate: PASS')"
```

## 2. 生成独立的 4060 真实对齐与全模态数据

先运行 Whisper 真实对齐。它使用音频词级时间戳承载标注文本，并用视频内嵌音轨对齐外部音频；精确字符覆盖率低于 50% 或音轨相关性不足会直接报错，不做均匀插值回退。`session_011` 已作为显式质量排除项写入入口；未匹配字符不分配时间戳，也不参与标签：

```powershell
.\scripts\data\prepare_whisper_alignment11_4060.ps1
```

中断后执行同一命令即可复用已经完成的 session；若明确需要全部重算，使用 `-Overwrite`。

### 2.1 已有旧 MARLIN cache 的快速对齐（推荐用于当前机器）

如果 `complete12/marlin` 下已经有旧的 12 个 MARLIN `.npz`，无需重新运行数小时的人脸检测和 MARLIN。下面的命令根据旧 embedding 在视频内的真实中心时间，将其最近邻重映射到“音频时间＋视频偏移”的 aligned11 时间轴：

```powershell
.\scripts\data\remap_marlin11_4060_cache.ps1
```

该步骤不使用 GPU，通常几十秒完成。生成的 11 个 cache 明确标记为 `legacy_embedding_time_remap_v1`，不会冒充重新逐帧提取。当前数据实测平均中心时间误差约 0.25–0.27 秒，最大约 0.55 秒。详细报告写入：

```text
data/cache/bci_subjects_ept_v6_marlin4060_aligned11/marlin/remap_report.json
```

然后生成最终数据。日志中 11 人都应显示 `MARLIN cache hit`，只需运行 WavLM、MacBERT、传感器整理和数据写入：

```powershell
.\scripts\data\prepare_current_session_registered_marlin11_4060.ps1
```

快速路线复用旧 MARLIN embedding，因此不会再次运行 FaceX-Zoo/MARLIN。若不生成 remap cache，则准备脚本会走直接路线：按对齐后的真实视频时间在每个一秒时间步采样 16 帧、逐帧运行 FaceX-Zoo，并重新运行冻结 MARLIN；直接路线在 4060 上预计约 5.5–7 小时。

数据写入中断后续跑。它会重建未完成的 processed 目录，但复用 remap 或直接 MARLIN cache：

```powershell
.\scripts\data\prepare_current_session_registered_marlin11_4060.ps1 -Overwrite
```

另开一个 PowerShell 查看进度：

```powershell
Set-Location F:\EPT-Net\code
.\scripts\data\watch_marlin12_4060_progress.ps1
```

## 3. 数据完成门禁

```powershell
$root = ".\data\processed\bci_subjects_ept_v6_marlin4060_aligned11"
if (-not (Test-Path "$root\_SUCCESS.json")) { throw "4060 dataset is incomplete" }
$summary = Get-Content "$root\dataset_summary.json" -Raw | ConvertFrom-Json
if ($summary.num_subjects -ne 11) { throw "Expected 11 aligned sessions after excluding session_011" }
if (($summary.split_sessions.train.Count + $summary.split_sessions.val.Count + $summary.split_sessions.test.Count) -ne 11) { throw "Unexpected split total" }
if (@($summary.split_sessions.train + $summary.split_sessions.val + $summary.split_sessions.test) -contains "session_011") { throw "Excluded session_011 leaked into a split" }
if ($summary.behavior_features.video_backend -ne "marlin" -or $summary.behavior_features.audio_backend -ne "wavlm" -or $summary.behavior_features.text_backend -ne "macbert") { throw "Behavior backend mismatch" }
Write-Host "RTX 4060 dataset gate: PASS"
```

检查六个模态全部开启：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe -c "from eptnet.config import load_config; from eptnet.train import validate_training_cohort; c=load_config('configs/eptnet_marlin11_aligned_seed42.yaml'); assert c['data']['dataset_name']=='bci_subjects_ept_v6_marlin4060_aligned11'; assert len(c['data']['expected_session_ids'])==11 and c['data']['excluded_session_ids']==['session_011']; assert (c['data']['video_dim'],c['data']['audio_dim'],c['data']['text_dim'])==(384,768,768); keys=('use_video','use_audio','use_text','use_eeg_time','use_eeg_spec','use_hr'); assert all(c['model'][k] for k in keys); print(validate_training_cohort(c)); print('RTX 4060 aligned11 training contract: PASS')"
```

正式训练入口还会自动执行同一项 cohort contract。日志中必须先出现：

```text
{"event":"cohort_contract","status":"passed","session_count":11,...}
```

该门禁检查 11 个指定 ID 与 train/validation/test 的并集完全相等、三个 split 互斥、`session_011` 未泄漏、每个人的 video/audio/text 有有效时间步，并且每个 session 都带完整 Whisper 对齐元数据。验证报告同时写入 `run_metadata.json`。

## 4. 标签方向与目标说话人审计

训练前先执行 fail-closed 审计。它逐 session 检查：

- 固定语义为 `0=deception`、`1=truth`，评估正类为 0；
- `positive_mask == target_mask AND label==0`；
- Whisper 对齐记录中的目标说话人与标注记录一致；
- tensor 内 target-valid 数量与 `dataset_summary.json` 一致。

```powershell
Set-Location F:\EPT-Net\code

.\scripts\audit\audit_label_speaker_direction.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_marlin11_aligned_windowed_seed42.yaml
```

通过后输出：

```text
results/audits/marlin11_label_speaker_direction.json
results/audits/marlin11_label_speaker_direction.md
```

该审计证明内部标签方向和说话人 provenance 一致，但不能替代数据所有者对原始高亮语义的人工确认。

## 5. Majority 与 Logistic 基线

下面的入口只用 train 拟合标准化和 Logistic 参数，只用 validation 选择阈值，最后一次性评估 test。不会使用 test 拟合特征标准化、参数或阈值：

```powershell
.\scripts\experiments\run_classical_baselines.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_marlin11_aligned_windowed_seed42.yaml `
  -Output results\baselines\marlin11_classical_seed42.json `
  -Device cpu
```

输出：

```text
results/baselines/marlin11_classical_seed42.json
```

当前真实运行参考值为 Majority Macro-F1 `0.4396`、Logistic Macro-F1 `0.4806`、Logistic AUROC `0.4624`。这些是用于判断神经网络是否真正超过简单模型的基线，不是主模型结果。

## 6. 128 秒因果窗口 EPT-Net（推荐新训练）

新协议只改变训练采样：

- 训练窗口长度 128 秒、步长 32 秒；
- 除每个 session 的第一个窗口外，窗口开头 32 秒只作为因果 warm-up，上下文可见但不计算任何 loss；
- 自动类别权重仍由 7 个训练 session 的唯一时间步计算，不按重叠窗口重复统计；
- validation 和 test 始终使用去重后的完整 session 时间线；
- 非窗口版配置 `eptnet_marlin11_aligned_seed42.yaml` 仍可独立使用；旧实验 ID 的 checkpoint 仅作历史归档，不与新配置混合续训。

先 smoke。它读取一个真实训练 session 的全部因果窗口和一个完整 validation session，不读取 test：

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_marlin11_aligned_windowed_seed42.yaml `
  -Seed 42 `
  -Device cuda:0 `
  -Smoke
```

正式训练：

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_marlin11_aligned_windowed_seed42.yaml `
  -Seed 42 `
  -Device cuda:0
```

中断后原样恢复：

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_marlin11_aligned_windowed_seed42.yaml `
  -Seed 42 `
  -Device cuda:0 `
  -Resume results\eptnet_marlin11_aligned_windowed_seed42\seed_42\last.pt
```

评估新模型：

```powershell
.\scripts\experiments\evaluate.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_marlin11_aligned_windowed_seed42.yaml `
  -Checkpoint results\eptnet_marlin11_aligned_windowed_seed42\seed_42\best.pt `
  -Device cuda:0
```

测试指标将写到同一个 `seed_42` 目录下的 `test_metrics.json`、`test_metrics_predictions.jsonl` 和 `test_metrics_events.json`。

## 7. 同协议 Early-Fusion GRU 基线

GRU使用完全相同的数据、128 秒窗口、warm-up、损失和完整 session 评估协议，只替换时序模型：

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\gru_marlin11_aligned_windowed_seed42.yaml `
  -Seed 42 `
  -Device cuda:0 `
  -Smoke

.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\gru_marlin11_aligned_windowed_seed42.yaml `
  -Seed 42 `
  -Device cuda:0

.\scripts\experiments\evaluate.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\gru_marlin11_aligned_windowed_seed42.yaml `
  -Checkpoint results\gru_marlin11_aligned_windowed_seed42\seed_42\best.pt `
  -Device cuda:0
```

## 8. 旧完整-session实验复现

先运行 smoke：

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_marlin11_aligned_seed42.yaml `
  -Seed 42 `
  -Device cuda:0 `
  -Smoke
```

Smoke 成功后执行正式训练，不要添加 `-Smoke`：

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_marlin11_aligned_seed42.yaml `
  -Seed 42 `
  -Device cuda:0
```

训练中断后恢复：

```powershell
.\scripts\experiments\train.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_marlin11_aligned_seed42.yaml `
  -Seed 42 `
  -Device cuda:0 `
  -Resume results\eptnet_marlin11_aligned_seed42\seed_42\last.pt
```

## 9. 旧完整-session模型最终评估

只读取训练选出的 `best.pt`；阈值只在 validation 上校准，然后冻结并评估 test：

```powershell
.\scripts\experiments\evaluate.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Config configs\eptnet_marlin11_aligned_seed42.yaml `
  -Checkpoint results\eptnet_marlin11_aligned_seed42\seed_42\best.pt `
  -Device cuda:0
```

不要把 4060 与 4090 的 processed dataset、checkpoint 或评估目录交叉使用。硬件路线分离只是为了运行隔离；正式比较模型时仍需使用相同数据版本和实验协议。
