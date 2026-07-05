# yolo-agent

中文 | [English](README.en.md)

YOLO Agent 是一个证据驱动的 YOLO 自动优化训练 harness。

它不是自由形式的代码生成 Agent，也不会盲目改模型代码。它把目标检测优化固定成一个可恢复、可审计的闭环：

```text
环境检查 -> debug 训练 -> 证据导入 -> 错误诊断 -> 下一轮优化建议
```

## 适合谁

- 想在 COCO 或自定义 YOLO 数据集上自动化训练和优化的人
- 想让实验有 queue、resume、evidence、report，而不是手动乱跑的人
- 想比较模型效果、延迟、模型大小和稳定性的工程团队

## 现在能做什么

- 一键检查 Python、CUDA、Ultralytics、COCO 路径、磁盘和 run 目录权限
- 一键启动 COCO + YOLO26 debug 训练，并在 debug 成功后自动进入 pilot
- 自动生成 run context、dataset manifest、experiment plan、execution queue 和 report
- 自动导入 `results.csv`、`best.pt`、`args.yaml`、runtime profile 和 COCO error facts
- 用 evidence gate、full-run 二次确认和 timeout 避免误跑大任务

## 安装

### Windows / PowerShell

```powershell
cd E:\codex\YOLO-Agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

### 可选：安装训练依赖

```powershell
python -m pip install -e ".[train]"
```

如果你已经单独安装了 Ultralytics，也可以只验证：

```powershell
python -c "import ultralytics; print(ultralytics.__version__)"
```

### 验证安装

```powershell
yolo-agent --help
yolo-agent doctor --data E:\dataset\coco.yaml --model yolo26n.pt
```

## 30 秒快速开始：COCO + YOLO26

1. 体检环境：

```powershell
yolo-agent doctor --data E:\dataset\coco.yaml --model yolo26n.pt
```

2. 启动自动优化训练。默认会先跑 debug；debug 成功后自动进入 pilot：

```powershell
yolo-agent optimize coco `
  --model yolo26n.pt `
  --data E:\dataset\coco.yaml `
  --goal +2map `
  --run-id coco-yolo26n `
  --profile debug `
  --execute
```

3. 查看训练状态和下一步建议：

```powershell
yolo-agent loop status --run runs/coco-yolo26n
```

4. 推进到 full COCO 时必须二次确认：

```powershell
yolo-agent optimize advance `
  --run runs/coco-yolo26n `
  --to-profile baseline_full `
  --execute `
  --confirm-full-run
```

## 自定义 YOLO 数据集

```powershell
yolo-agent optimize custom `
  --model yolo26n.pt `
  --data path\to\data.yaml `
  --run-id custom-yolo26n `
  --profile debug `
  --execute
```

`data.yaml` 使用标准 YOLO 格式。先跑 `debug`，确认路径、类别和最小训练流程没问题，再升级到 `pilot` 或 full profile。

## 安全边界

- 默认只做 dry-run；只有显式加 `--execute` 才会启动训练
- `debug` 是小比例、短训练的 sanity run；debug 成功后默认自动进入 `pilot`
- 需要只跑当前 profile 时可以加 `--no-auto-advance`
- `baseline_full`、`baseline_confirm`、`candidate_full` 都必须额外加 `--confirm-full-run`
- debug 默认 timeout 为 3600 秒，pilot 默认 timeout 为 43200 秒
- 自动推进是有限状态：`debug -> pilot`，只有显式确认 full run 后才允许继续 full profile；不会无限循环

## 常用命令

```powershell
yolo-agent doctor --data E:\dataset\coco.yaml --model yolo26n.pt
yolo-agent optimize coco --model yolo26n.pt --data E:\dataset\coco.yaml --run-id coco-yolo26n --profile debug --execute
yolo-agent loop status --run runs/coco-yolo26n
yolo-agent report --run runs/coco-yolo26n --out report.md
```

## 文档导航

- [安装指南](docs/install.md)
- [快速开始](docs/quickstart.md)
- [COCO + YOLO26 Runbook](docs/coco-yolo26.md)
- [自定义数据集](docs/custom-dataset.md)
- [核心概念](docs/concepts.md)
- [Loop Engineering](docs/loop-engineering.md)
- [Evidence 和报告](docs/evidence.md)
- [CLI 参考](docs/cli.md)
- [故障排查](docs/troubleshooting.md)

## 开发

```powershell
python -m pip install -e ".[dev]"
py -3.12 -m pytest
```

## 项目定位

YOLO Agent is a componentized object-detection optimization harness, not a free-form code-generation agent.
