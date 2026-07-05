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
yolo-agent optimize advance --run runs/coco-yolo26n --to-profile baseline_full --execute --confirm-full-run
```

## 训练太慢

先看状态面板：

```powershell
yolo-agent loop status --run runs/coco-yolo26n
```

重点检查 GPU util、it/s、batch size、cache mode、dataloader wait。不要先改 `imgsz`，否则 baseline 不可比。

