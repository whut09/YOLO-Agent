# 安装指南

本文面向 Windows / PowerShell 用户。Linux 和 macOS 也可以使用同样的 `pip install -e` 流程。

## 1. 准备 Python

推荐 Python 3.10+。当前开发环境使用 Python 3.12：

```powershell
py -3.12 --version
```

## 2. 创建虚拟环境

```powershell
cd E:\codex\YOLO-Agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

如果 PowerShell 阻止激活脚本，可以先执行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. 安装项目

只做开发和 dry-run：

```powershell
python -m pip install -e ".[dev]"
```

需要真实训练：

```powershell
python -m pip install -e ".[train]"
```

## 4. 验证命令

```powershell
yolo-agent --help
python -m pytest
```

## 5. 检查训练环境

```powershell
yolo-agent doctor --data E:\dataset\coco.yaml --model yolo26n.pt
```

`doctor` 会检查 Python、Ultralytics、CUDA driver、PyTorch CUDA、可用显存、COCO 路径、annotations、磁盘空间和 run 目录权限。

