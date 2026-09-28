# EPT-Net 服务器运行手册（11-session 全模态）

本入口用于 Linux CUDA 服务器，直接读取已经完成真实时间对齐和冻结特征提取的 11 个 session。它不会重新运行 Whisper、FaceX-Zoo、MARLIN、WavLM 或 MacBERT。训练使用视频、音频、文本、EEG 时域、EEG 频域和 HR 六个模态，并强制使用 `cuda:0`；GPU 不可见时立即退出，不会退回 CPU。

## 目录约定

```text
/root/EPT2026/                                      # Git 项目
/root/EPT2026/data                                 # 软链接
/root/autodl-tmp/.autodl/data/                     # 服务器大文件根目录
  processed/bci_subjects_ept_v6_marlin4060_aligned11/
    _SUCCESS.json
    dataset_summary.json
    manifests/sessions_train.jsonl
    manifests/sessions_val.jsonl
    manifests/sessions_test.jsonl
    samples/*.pt
```

训练和评估会生成但不会提交到 Git：

```text
results/eptnet_v6_marlin11_4060_windowed_seed42/seed_42/
results/baseline_early_fusion_gru_marlin11_4060_windowed_seed42/seed_42/
results/baselines/marlin11_classical_seed42.json
logs/
fig/fig01-training-dynamics/*.{pdf,svg,png}
fig/fig02-model-comparison/*.{pdf,svg,png}
fig/fig03-heldout-uncertainty/*.{pdf,svg,png}
```

## 1. 拉取代码并连接已有数据

```bash
cd /root/EPT2026
git status --short
git pull --ff-only origin main

test -d /root/autodl-tmp/.autodl/data
if [ -e data ] && [ ! -L data ]; then
  echo "data exists but is not a symlink; stop and inspect it" >&2
  exit 1
fi
ln -sfn /root/autodl-tmp/.autodl/data data
```

不要把大文件复制进 Git 工作树；`data/` 只作为上述软链接存在。

## 2. 环境安装

CUDA 版 PyTorch 应与服务器驱动匹配。当前服务器已安装 `torch 2.8.0+cu128` 时，只需安装锁定依赖和项目本身：

```bash
cd /root/EPT2026
/root/miniconda3/bin/python -m pip install -r requirements-lock.txt
/root/miniconda3/bin/python -m pip install -e .
/root/miniconda3/bin/python -m pip check
```

论文图严格要求 Arial，不允许静默字体替换。Ubuntu/Debian 可安装 Microsoft core fonts：

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y fontconfig ttf-mscorefonts-installer
fc-cache -f
/root/miniconda3/bin/python - <<'PY'
from matplotlib import font_manager
p = font_manager.findfont("Arial", fallback_to_default=False)
assert font_manager.FontProperties(fname=p).get_name() == "Arial", p
print("Arial:", p)
PY
```

若镜像源不提供该包，请把合法获得的 `arial.ttf`/`arialbd.ttf` 放入 `/root/.local/share/fonts/arial/`，执行 `fc-cache -f` 后重跑字体门禁。

## 3. GPU、数据和六模态门禁

```bash
cd /root/EPT2026
chmod +x scripts/server/verify_marlin11_gpu.sh scripts/experiments/run_marlin11_server.sh
./scripts/experiments/run_marlin11_server.sh check
```

通过时会输出 `status=PASS`、实际 GPU 名称、CUDA 版本、11 人 cohort 和六个已启用模态。以下任一情况都会阻止训练：GPU 未挂载、CUDA 不可用、数据集没有 `_SUCCESS.json`、session 数量不是 11、任一模态被关闭或 split/cohort 不一致。

可额外观察 GPU：

```bash
watch -n 1 nvidia-smi
```

## 4. 只做最小 smoke（当前建议）

这一步不属于正式实验，只验证一轮真实数据、反向传播、CUDA 和写盘链路：

```bash
cd /root/EPT2026
./scripts/experiments/run_marlin11_server.sh smoke-main
```

若同名 smoke 目录已经存在，训练器会保护旧产物并拒绝覆盖。先检查其内容；确认是可删除的 smoke 产物后再执行：

```bash
rm -rf -- /root/EPT2026/results/smoke/eptnet_v6_marlin11_4060_windowed_seed42_smoke
./scripts/experiments/run_marlin11_server.sh smoke-main
```

不要在 GPU 门禁失败时启动完整实验。

## 5. 单独运行主模型

首次训练：

```bash
cd /root/EPT2026
./scripts/experiments/run_marlin11_server.sh train-main
```

中断后续跑（读取可信的 `last.pt`）：

```bash
cd /root/EPT2026
./scripts/experiments/run_marlin11_server.sh resume-main
```

训练完成后评估最佳 checkpoint：

```bash
./scripts/experiments/run_marlin11_server.sh eval-main
```

## 6. 完整实验命令（暂不执行）

`full` 会依次完成：GPU/数据门禁 → Majority/Logistic → EPT-Net 训练与评估 → matched Early-Fusion GRU 训练与评估 → 三幅论文图。若主模型或 GRU 已有 `last.pt`，它会自动续跑；若已有 `test_metrics.json`，它会跳过该完成项。

```bash
cd /root/EPT2026
nohup ./scripts/experiments/run_marlin11_server.sh full \
  > logs/marlin11_full_seed42.log 2>&1 &
echo $! > logs/marlin11_full_seed42.pid
tail -f logs/marlin11_full_seed42.log
```

停止跟踪日志用 `Ctrl+C`，不会终止后台训练。检查进程和 GPU：

```bash
cat logs/marlin11_full_seed42.pid
ps -fp "$(cat logs/marlin11_full_seed42.pid)"
nvidia-smi
```

## 7. 单独运行 matched GRU 和经典基线

```bash
cd /root/EPT2026
./scripts/experiments/run_marlin11_server.sh classical
./scripts/experiments/run_marlin11_server.sh smoke-gru
./scripts/experiments/run_marlin11_server.sh train-gru
./scripts/experiments/run_marlin11_server.sh eval-gru
```

GRU 中断后改用：

```bash
./scripts/experiments/run_marlin11_server.sh resume-gru
```

## 8. 论文级图件

绘图脚本只读取真实的 `history.json`、`test_metrics.json` 和经典基线 JSON；缺失任何输入都会失败，不会生成模拟数值。三幅图分别展示训练动态、四模型 test 指标比较、 held-out subject bootstrap 置信区间。

```bash
cd /root/EPT2026
./scripts/experiments/run_marlin11_server.sh figures
```

每幅图都有独立的 `figure.yaml`、独立脚本和 `qa-report.json`，并从同一画布同步导出：

- PDF：矢量版；
- SVG：可编辑矢量版；
- PNG：500 dpi；
- QA：尺寸、像素、字体、颜色、输入文件 SHA-256 和数据范围检查。

自动 QA 中 `visual_preview` 保持 `manual-required`，因为最终裁切、重叠和视觉平衡必须人工查看 PNG，不能伪装成自动通过。

## 9. 清理规则

源码仓库只保留运行、验证、复现和说明所需文件。以下目录属于可再生产物且已被 `.gitignore` 排除：

```text
data/  results/  logs/  figures/
fig/**/*.pdf  fig/**/*.svg  fig/**/*.png  fig/**/qa-report.json
```

清理 Python 缓存不会影响数据或 checkpoint：

```bash
cd /root/EPT2026
find src tests scripts fig -type d -name __pycache__ -prune -exec rm -rf -- {} +
rm -rf -- .pytest_cache .mypy_cache .ruff_cache src/eptnet.egg-info
```

不要清理 `/root/autodl-tmp/.autodl/data`，不要删除正在续跑的 `results/**/last.pt`，也不要重新运行数据对齐或视频特征提取脚本。
