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

结构化目标和自然语言意图是两个字段：

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-small --target-metric ap_small --target-delta 0.02 --goal-description "Reduce small-object false negatives"
```

- `--goal`：`+2map`、`+0.02map50_95`、`+2ppmap50` 或 `+2%map`。
- `--target-metric` 与 `--target-delta`：显式指标和归一化绝对增益，必须同时提供。
- `--goal-description`：只作为诊断意图保存，不替代可执行目标。

常见问题可以直接写成人话，Agent 负责选择算法和已认证 recipe：

| 问题 | 推荐验收目标 | 描述示例 |
| --- | --- | --- |
| 误检多 | `--target-metric precision --target-delta 0.02` | `--goal-description "当前模型误检多，请降低高置信度误检"` |
| 新场景效果差 | `--target-metric map50_95 --target-delta 0.02` | `--goal-description "换到新场景后效果下降，请诊断场景偏移"` |
| 小目标漏检 | `--target-metric ap_small --target-delta 0.02` | `--goal-description "提高 AP_small 并减少小目标漏检"` |
| 提高整体 mAP | `--goal +2map` | `--goal-description "提高整体 mAP，同时控制延迟和模型大小"` |

新场景优化必须提供能代表目标场景的训练和验证数据。描述用于诊断与 recipe 选择，结构化目标用于确定性验收；省略结构化目标时默认使用 `+2map`。系统会自动尝试并比较候选，但不保证每次运行都能提高 mAP。

所有目标参数会在 run-id 分配和目录创建前验证。若已有同名目录但没有
`run_context.yaml`，系统保留该目录、生成
`artifacts/run_initialization_migration.yaml`，并使用递增的新 run-id。

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
yolo-agent research build-snapshot --root research --source awesome_object_detection --maturity-registry runs/component_maturity_registry.yaml
```

可选的本地代码元数据缓存：

```powershell
yolo-agent research build-snapshot --root research --source awesome_object_detection --cached-code-root E:\paper-code-cache
```

该参数只读取本地 README/config，不联网，也不会将 paper claim 转换成本地 evidence。

`--maturity-registry` 默认就是 `runs/component_maturity_registry.yaml`。snapshot 会冻结 MethodProfile coverage、有效 adapter hash、Ultralytics 版本、认证 protocol 和 maturity artifacts。

真实 `train` 会在分配 run-id 前检查 snapshot。缺失、损坏或 stale 时，终端打印可直接执行的 `build-snapshot` 命令且不创建 run。训练期间不会执行 catalog importer、PaperScout、live maturity overlay 或网络请求；已有 run 绑定创建时的 snapshot 身份。

## Advanced：GPU 认证

GPU certification 是显式、opt-in 的验证流程：

```powershell
yolo-agent advanced certify-gpu --help
```

它用于验证 adapter、matched pilot、post-eval、paired delta、ASHA 和多种子确认链路。默认测试与默认训练不会自动运行 full COCO；full COCO 必须由当前 objective、dataset manifest 和预算范围内的显式确认授权。

单个组件在进入 matched pilot 前，应先完成运行时认证：

```powershell
yolo-agent advanced certify-component --component sampling.small_object --cpu
yolo-agent advanced certify-component --component sampling.small_object --gpu --device 0
yolo-agent advanced certify-paper-components --model E:\path\yolo26n.pt --teacher E:\path\yolo26s.pt --device 0 --execute-real-gpu
yolo-agent advanced certify-component --component loss.quality.correlation --cpu
yolo-agent advanced certify-component --component loss.calibration.bpc --cpu
yolo-agent advanced certify-component --component loss.quality.pseudo_iou --cpu
yolo-agent advanced certify-component --component distillation.yolo26_teacher_student --cpu
yolo-agent advanced certify-component --component distillation.feature --cpu
yolo-agent advanced certify-component --component head.p2_small_object --cpu
yolo-agent advanced certify-component --component neck.multi_scale_fusion --cpu
yolo-agent advanced certify-component --component assigner.task_aligned --cpu
yolo-agent advanced certify-component --component assigner.dynamic_topk --cpu
yolo-agent advanced certify-component --component assigner.dual_path --cpu
yolo-agent advanced certify-paper-adapters --cpu
yolo-agent advanced certify-paper-adapters --cpu --resume
yolo-agent advanced certify-paper-adapters --cpu --changed-only
yolo-agent advanced certify-paper-adapters --gpu --execute-real-gpu --model yolo26n.pt --data coco.yaml --device 0
```

四个高价值机制完成 `gpu_certified` 后，可运行论文自动优化验收：

```powershell
yolo-agent advanced certify-paper-auto --workdir runs/certification/paper-auto --research-root research --source E:\path\Awesome-object-detection --registry runs/component_maturity_registry.yaml --policy-root runs --model yolo26n.pt --device 0 --execute-real-gpu
```

该命令构建 sampling、auxiliary loss、distillation、model graph 四个 family 的
matched `pilot_3` cohort，并且只执行 ASHA 签发的 `pilot_10` survivor。缺少 post-eval、
error facts 或 paired bootstrap 时立即进入 evidence recovery。它不会运行 full 或 seed
2/3；这些预算仍要求训练流程中的显式 `--confirm-full-run`。

Assignment CPU certification is shadow-only. Active execution requires the generated
same-protocol shadow artifact, a matched control, and ASHA materialization.

Loss and distillation CPU certification uses the real Ultralytics trainer plugin
bridge and writes an independent golden-path artifact for each AtomicRecipe. GPU mode
is explicit opt-in and requires the matching CPU `smoke_passed` registry overlay.

P2 和三个 neck 的 CPU 认证会检查真实 graph、native loss、backward、AMP、
partial checkpoint、export 和资源门禁。TOOD-TAL、OTA、DSLA assignment 认证只运行
shadow mode，记录 positive ratio 与 conflict rate，并验证 native loss 完全不变。
assignment active pilot 还必须通过同 protocol shadow artifact 与 matched control 门禁；
组件认证不会直接创建训练任务。

`--cpu` 在隔离进程中依次验证 adapter import、runtime payload、Ultralytics hook 签名、单元检查和 smoke，只能将有效本地证据推进到 `smoke_passed`。`--gpu` 本身是显式 GPU 授权，并且只在同一 registry、protocol 和代码版本下已有有效 CPU `smoke_passed` 时运行；组件没有实现 `gpu_smoke_test` 时会 fail closed，不会退化成普通 Ultralytics 训练。

Distillation 支持 `logits`、`feature`、`localization`、`relation`、`attention`、`masked_feature`、`quality_aware` 和 `teacher_ensemble` 独立组件。GPU 认证必须提供本地 teacher；ensemble 还必须通过 `--ensemble-teacher` 显式提供第二个不同 checkpoint，认证命令不会下载模型或猜测路径。

默认生成目录是 `runs/certification/components/<component-id>`，本机证据写入 `runs/component_maturity_registry.yaml`。终端会打印当前成熟度、缺失 artifact、生成路径、失败原因和下一成熟度。修改 adapter、Ultralytics 版本或 protocol 后，旧证据会自动失效。

SAHI 切片属于独立推理认证，不属于训练 recipe：

```powershell
yolo-agent advanced certify-sahi --help
```

它只输出 `sliced_*` 指标并进入独立 Pareto front，不覆盖标准 640 指标，也不把推理收益归因到训练组件。详见 [SAHI Independent Inference Certification](sahi-inference-certification.md)。

其他 inference-only 论文策略使用统一高级命令：

```text
yolo-agent advanced certify-inference-policy --help
```

该命令接收结构化 policy YAML，支持 tiled multi-scale、TTA、置信度校准、
按类阈值和跨视图 merge。每种策略保留独立指标 namespace 和 Pareto front，
不会创建训练 recipe。详见 [Isolated Inference Policy Adapters](inference-policy-adapters.md)。

## 内部兼容命令

项目可能保留 doctor、队列、证据、复现和旧 optimize 子命令，供测试、迁移和维护使用。它们不是稳定的新手接口，也不应出现在普通运行的 `Next:` 提示中。

更多背景见 [训练模式](training-modes.md)、[Paper Intelligence](paper-intelligence.md)、[Component Maturity Registry](component-maturity-registry.md) 和 [GPU Certification](gpu-certification.md)。
