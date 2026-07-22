# YOLO Agent

中文 | [English](README.en.md)

YOLO Agent 是一个面向 YOLO 目标检测的证据驱动自动优化训练工具。它把训练、COCO 评估、错误诊断、候选配方、预算淘汰和报告串成可恢复、可审计的闭环；LLM 可以提出建议，但不能绕过兼容性、证据、预算、ASHA 和 full-run 确认门禁。

## 安装

建议使用 Python 3.12 和独立虚拟环境：

```powershell
cd E:\codex\YOLO-Agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[train]"
```

## 三步启动：一条命令开始训练

新人只需要记住四个命令：`setup`、`train`、`status` 和 `stop`。

1. 检查环境并生成本地配置：

```powershell
yolo-agent setup coco --data E:\datatset\coco.yaml --model yolo26n.pt
```

2. 启动自动训练与 pilot 优化：

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n
```

3. 查看聚合状态：

```powershell
yolo-agent status --run runs/coco-yolo26n
```

需要停止时运行：

```powershell
yolo-agent stop --run runs/coco-yolo26n
```

## 自动优化会做什么

### 运行模式一句话

默认从环境检查进入 debug/pilot 自动闭环，在证据、预算或停止条件触发时结束；full COCO 不会默认启动。

默认流程读取可信 baseline/pilot evidence，完成固定协议的评估和 error facts，结合规则、历史策略与冻结的论文快照生成候选，再由确定性门禁检查组件成熟度、YOLO26 兼容性、`imgsz=640`、公平对照和预算。通过的候选进入 ASHA 管理的 pilot 队列，训练后导入 paired delta、延迟和模型大小证据，并自动决定淘汰、补证据或继续。

默认流程不会自动增加输入尺寸，也不会默认启动 full COCO。达到 full 候选阶段后必须显式确认；`+2 mAP` 是优化目标，不是自动保证。

## 论文库是什么

项目可以离线导入 [Awesome-object-detection](https://github.com/whut09/Awesome-object-detection)，并在训练前生成冻结的 `ResearchSnapshot`。训练期间只读取该快照，不联网读取论文。

论文库不是训练集，论文指标也不是本地 evidence。论文记录只提供诊断和 recipe prior：

- `recipe_idea_only` 不是可执行 recipe。
- 有论文记录不代表已有 adapter。
- 有 adapter 不代表已经 smoke passed。
- smoke passed 不代表 pilot reproduced。
- pilot reproduced 不代表 full COCO confirmed。

## 仍需本地认证

真实 adapter 必须经过构造、shape、backward、AMP 和 smoke 测试；候选收益还需要 matched pilot、完整 COCO evidence、延迟/模型大小 guard 和多种子确认。只有本机认证产物可以提升本地复现状态，论文 claim、代码存在或一次单 seed 提升都不能替代认证。

<!-- capability-maturity:start -->
| 能力 | 当前状态 | 代码存在 | 自动执行 | 本地复现 | 现实边界 |
| --- | --- | --- | --- | --- | --- |
| Pilot 自动训练 | `executable` | 是 | 是 | 取决于本地 run | 默认训练入口可执行 debug/pilot；是否成功取决于本机环境和数据。 |
| 自动导入基础指标 | `executable` | 是 | 是 | 取决于本地 run | 可导入 results.csv、训练 artifacts 和基础 runtime evidence；缺失产物仍会形成 evidence gap。 |
| Candidate COCO error facts | `incomplete` | 是 | 部分 | 部分 | 已有 post-eval、导入和 completeness gate，但每个候选都稳定产出 predictions.json 与完整 per-class/FN/FP/localization facts 的闭环尚未完全保证。 |
| Error-delta 下一轮决策 | `partial` | 是 | 部分 | 部分 | 能比较 parent/current error facts 并约束 proposal；候选 error facts 不完整时会退回补证据或规则路径。 |
| ASHA / successive halving 队列控制 | `executable` | 是 | 有门禁 | 未声明 | ASHA assignment 已进入权威 RoundExecutionPlan 和队列；full rung 仍必须显式确认，不能理解为默认自动跑完整 COCO。 |
| 论文组件 Adapter | `mixed` | 是 | 混合 | 混合 | registry 同时包含 metadata-only、已实现 adapter 和可执行组件；必须逐组件查看 maturity，不能把论文条目等同于可训练实现。 |
| 3-seed confirmation | `supported, not automatic end-to-end` | 是 | 需显式确认 | 未声明 | 调度器和 confidence gate 支持 3 seeds；candidate_full 需要显式 full 确认，默认 pilot loop 不会自动完成全部 seeds。 |
| 稳定提升 +2 mAP | `not guaranteed` | 否 | 否 | 未声明 | +2 mAP 是优化目标和验收条件，不是项目保证；必须由 matched baseline、full COCO、3 seeds 和置信区间证明。 |
<!-- capability-maturity:end -->

## 下一步读哪个文档

- [命令与高级入口](docs/cli.md)
- [Awesome-object-detection 离线适配](docs/awesome-object-detection.md)
- [Paper Intelligence](docs/paper-intelligence.md)
- [能力成熟度](docs/capability-maturity.md)
- [GPU Certification](docs/gpu-certification.md)
- 新人专题：[安装](docs/install.md)、[快速开始](docs/quickstart.md)、[训练模式](docs/training-modes.md)、[COCO + YOLO26](docs/coco-yolo26.md)、[自定义数据集](docs/custom-dataset.md)、[LLM 设置](docs/llm-setup.md)、[故障排查](docs/troubleshooting.md)
