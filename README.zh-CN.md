# YOLO Agent

[English](README.md) | 简体中文

YOLO Agent 是一个面向 YOLO 目标检测的证据驱动自动优化训练工具。它将训练、COCO 评估、错误诊断、论文方法建议、公平对照、预算淘汰和报告整合为可恢复、可审计的工作流。

LLM 可以分析证据并提出 recipe，但兼容性、实验预算、晋级和 full-run 确认始终由确定性门禁控制。

![YOLO Agent 架构图](docs/assets/yolo-agent-architecture.svg)

## 核心能力

- 一条命令完成环境检查、debug 训练和有预算边界的 pilot 优化。
- 使用 matched baseline、本地 evidence、延迟和模型大小决定候选去留。
- 由 ASHA 管理 pilot 预算，尽早淘汰无效候选。
- Paper Intelligence 可离线导入论文目录并冻结为 `ResearchSnapshot`。
- 组件成熟度门禁阻止 metadata-only 或未验证 adapter 进入训练。
- 论文 recipe 必须绑定通过校验的 runtime adapter 和 matched control，ASHA 才会分配 pilot。
- 每个 run 都保存计划、事件、证据、队列状态和报告。

## 安装

建议使用 Python 3.12 和独立虚拟环境。

```powershell
git clone https://github.com/whut09/YOLO-Agent.git
cd YOLO-Agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[train]"
```

如需 GPU 认证和完整 COCO post-eval 支持：

```powershell
python -m pip install -e ".[train,certification]"
```

## 快速开始

新人只需要四个命令。

```powershell
# 1. 检查环境
yolo-agent setup coco --data E:\dataset\coco.yaml --model yolo26n.pt

# 2. 启动有预算边界的自动 pilot 优化
yolo-agent train --model yolo26n.pt --data E:\dataset\coco.yaml --run-id coco-yolo26n

# 3. 查看基础 run 和当前 child run 的聚合状态
yolo-agent status --run runs/coco-yolo26n

# 4. 安全停止
yolo-agent stop --run runs/coco-yolo26n
```

默认使用自动预算并只运行 pilot。Full COCO 训练必须显式确认。

## 决策流程

```text
可信 baseline 和当前 evidence
-> COCO error facts 与诊断
-> 论文及本地 recipe 建议
-> 兼容性、成熟度和证据门禁
-> matched pilot cohort
-> post-eval 与 paired delta
-> ASHA 淘汰或晋级
-> 报告与 policy memory 更新
```

关键证据不完整时，队列只会请求补证据，不会继续晋级训练。YOLO26 公平对比固定使用 `imgsz=640`，系统不会自动增加输入尺寸。

## Paper Intelligence

YOLO Agent 可以在训练前离线导入 [Awesome-object-detection](https://github.com/whut09/Awesome-object-detection)，并生成冻结的研究快照。训练期间不会联网读取论文。

论文记录只是先验，不是本地训练结果：

- 有论文记录不代表已有 adapter。
- adapter 已实现不代表运行链路和 smoke test 已通过。
- 单次 pilot 提升只能标记为 `possible`，不能写成 `confirmed`。
- 论文指标不能作为候选晋级证据。

<!-- paper-adapter-coverage:start -->
| 冻结论文 | 已实现 adapter | Runtime integrated | Pilot reproduced |
| --- | --- | --- | --- |
| 728 | 13 | 0 | 0 |

这些计数相互独立；论文记录和 adapter 类不会自动提升运行或复现成熟度。
Audit snapshot: `c606d6c50fefaa7ae0db8bddb39d62057ff09ed5aeae943c81c990971b353e57`.
<!-- paper-adapter-coverage:end -->

## 能力边界

<!-- capability-maturity:start -->
| 能力 | 当前状态 | 代码存在 | 自动执行 | 本地复现 | 现实边界 |
| --- | --- | --- | --- | --- | --- |
| Pilot 自动训练 | `executable` | 是 | 是 | 取决于本地 run | 默认训练入口可执行 debug/pilot；是否成功取决于本机环境和数据。 |
| 自动导入基础指标 | `executable` | 是 | 是 | 取决于本地 run | 可导入 results.csv、训练 artifacts 和基础 runtime evidence；缺失产物仍会形成 evidence gap。 |
| Candidate COCO error facts | `incomplete` | 是 | 部分 | 部分 | 已有 post-eval、导入和 completeness gate，但每个候选都稳定产出 predictions.json 与完整 per-class/FN/FP/localization facts 的闭环尚未完全保证。 |
| Error-delta 下一轮决策 | `partial` | 是 | 部分 | 部分 | 能比较 parent/current error facts 并约束 proposal；候选 error facts 不完整时会退回补证据或规则路径。 |
| ASHA / successive halving 队列控制 | `executable` | 是 | 有门禁 | 未声明 | ASHA assignment 已进入权威 RoundExecutionPlan 和队列；full rung 仍必须显式确认，不能理解为默认自动跑完整 COCO。 |
| 论文组件 Adapter | `incomplete` | 是 | 否 | 未声明 | 当前有 13 个 adapter 实现，但没有组件具备 artifact-backed runtime integration 或 pilot reproduction；论文条目不能进入训练队列。 |
| 3-seed confirmation | `supported, not automatic end-to-end` | 是 | 需显式确认 | 未声明 | 调度器和 confidence gate 支持 3 seeds；candidate_full 需要显式 full 确认，默认 pilot loop 不会自动完成全部 seeds。 |
| 稳定提升 +2 mAP | `not guaranteed` | 否 | 否 | 未声明 | +2 mAP 是优化目标和验收条件，不是项目保证；必须由 matched baseline、full COCO、3 seeds 和置信区间证明。 |
<!-- capability-maturity:end -->

## 文档

- [快速开始](docs/quickstart.md)
- [安装](docs/install.md)
- [CLI 与高级命令](docs/cli.md)
- [训练模式](docs/training-modes.md)
- [COCO 与 YOLO26](docs/coco-yolo26.md)
- [自定义数据集](docs/custom-dataset.md)
- [LLM 设置](docs/llm-setup.md)
- [证据模型](docs/evidence.md)
- [Paper Intelligence](docs/paper-intelligence.md)
- [论文 Recipe 执行门禁](docs/paper-recipe-materialization.md)
- [Awesome-object-detection 适配](docs/awesome-object-detection.md)
- [能力成熟度](docs/capability-maturity.md)
- [GPU Certification](docs/gpu-certification.md)
- [故障排查](docs/troubleshooting.md)

## 开发

```powershell
python -m pip install -e ".[dev]"
pytest -q
ruff check .
```

本项目采用 MIT License。
