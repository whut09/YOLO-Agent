# 故障排查

## yolo-agent 命令不存在

确认虚拟环境已激活，并重新安装：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
yolo-agent --help
```

## PowerShell 不允许激活 venv

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## doctor 提示 Ultralytics 缺失

```powershell
python -m pip install -e ".[train]"
```

或：

```powershell
python -m pip install ultralytics
```

## doctor 提示 GPU 不可见

检查：

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

如果 `nvidia-smi` 不存在，先安装或修复 NVIDIA driver。

## PowerShell 显示 `>>` 没反应

这不是后台训练，也不是卡死。`>>` 是 PowerShell 的续行提示，表示上一行命令还没结束，通常是复制了反引号 `` ` ``，或反引号后有不可见空格。

先按 `Ctrl+C` 退出 `>>`，然后用单行命令：

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --goal +2map --run-id coco-yolo26n
```

## `Unsupported --goal expression`

`--goal` 不是自然语言字段，只接受 `+2map`、`+0.02map50_95`、
`+2ppmap50` 或 `+2%map`。例如小目标目标应写为：

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-small --target-metric ap_small --target-delta 0.02 --goal-description "Improve AP_small and reduce false negatives"
```

错误输入会在创建 run 前停止并打印可执行的 `Next:` 命令，不会输出 traceback。

## `research snapshot preflight failed`

真实训练只接受 current v7 snapshot。缺少 snapshot、artifact hash 损坏、旧 snapshot
缺少 `paper_method_coverage.yaml`，或有效 maturity 清单缺失时，系统不会创建 run。
直接执行终端打印的命令，默认形式为：

```powershell
yolo-agent research build-snapshot --root research --source awesome_object_detection --maturity-registry runs/component_maturity_registry.yaml
```

认证新组件、修改 adapter 或升级 Ultralytics 后也必须重建 snapshot，并使用新的
run-id。旧 run 继续绑定原 snapshot，不会读取更新后的 live overlay。

## 同名 run 自动增加编号

这是防止旧 protocol、queue 或半初始化目录污染新实验的预期行为。若旧目录没有
`run_context.yaml`，检查：

```text
runs/<requested-run-id>/artifacts/run_initialization_migration.yaml
```

系统不会自动删除旧目录；新实验使用 `<requested-run-id>-1` 等递增编号。

## COCO 路径不对

确认 `data.yaml` 的 `path` 指向数据集根目录，且至少包含：

```text
images/train2017
images/val2017
annotations/instances_val2017.json
```

## full COCO 被拦截

这是预期行为。full profile 必须显式加：

```powershell
--confirm-full-run
```

例如：

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n --profile baseline_full --confirm-full-run
```

## 训练太慢

先看状态面板：

```powershell
yolo-agent status --run runs/coco-yolo26n
```

重点检查 GPU util、it/s、batch size、cache mode、dataloader wait。不要先改 `imgsz`，否则 baseline 不可比。
