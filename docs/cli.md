# 命令行入口

YOLO Agent 把新人入口稳定在四个命令。日常训练不需要记忆内部队列、证据导入、论文同步或复现状态命令。

## 新人命令

### 1. setup

检查 Python、训练依赖、数据路径、GPU 和 batch 能力，并生成本地配置：

```powershell
yolo-agent setup coco --data E:\datatset\coco.yaml --model yolo26n.pt
```

### 2. train

统一训练入口。相同命令负责新建 run、恢复 run、继续自动 pilot loop 和读取已有状态：

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n
```

默认使用自动预算，固定公平对比输入尺寸 `imgsz=640`，并在 full COCO 前停止等待显式确认。不要用内部子命令手工推进普通训练。

### 3. status

读取 base run，并自动聚合当前 child run、阶段、训练进度、诊断、recipe、delta、剩余候选和下一步：

```powershell
yolo-agent status --run runs\coco-yolo26n
```

### 4. stop

请求训练循环在安全边界停止：

```powershell
yolo-agent stop --run runs\coco-yolo26n
```

终端中的 `Next:` 只应提示继续使用 `yolo-agent train ...`，或说明系统将自动继续；不会要求新人调用内部推进命令。

## Advanced：论文研究

研究命令用于训练前准备离线论文快照，不属于新人训练流程：

```powershell
yolo-agent research import-awesome --source E:\path\Awesome-object-detection
yolo-agent research import-awesome --source E:\path\Awesome-object-detection --dry-run
yolo-agent research build-snapshot --root research --source awesome_object_detection
```

训练期间不会执行 catalog importer、PaperScout 或网络请求。已有 run 绑定创建时的 `snapshot_hash`，live registry 后续变化不会悄悄改变该 run 的决策上下文。

## Advanced：GPU 认证

GPU certification 是显式、opt-in 的验证流程：

```powershell
yolo-agent advanced certify-gpu --help
```

它用于验证 adapter、matched pilot、post-eval、paired delta、ASHA 和多种子确认链路。默认测试与默认训练不会自动运行 full COCO；full COCO 必须由当前 objective、dataset manifest 和预算范围内的显式确认授权。

## 内部兼容命令

项目可能保留 doctor、队列、证据、复现和旧 optimize 子命令，供测试、迁移和维护使用。它们不是稳定的新手接口，也不应出现在普通运行的 `Next:` 提示中。

更多背景见 [训练模式](training-modes.md)、[Paper Intelligence](paper-intelligence.md) 和 [GPU Certification](gpu-certification.md)。
