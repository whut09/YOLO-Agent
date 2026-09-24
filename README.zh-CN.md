# YOLO Agent

[English](README.md) | 简体中文

YOLO Agent 是一个面向 YOLO 目标检测的证据驱动自动优化训练工具。它将训练、COCO 评估、错误诊断、论文方法建议、公平对照、预算淘汰和报告整合为可恢复、可审计的工作流。

LLM 可以分析证据并提出 recipe，但兼容性、实验预算、晋级和 full-run 确认始终由确定性门禁控制。

![YOLO Agent 架构图](docs/assets/yolo-agent-architecture.svg)

> 本文所有命令示例均为单行 PowerShell 写法。PowerShell 不支持 Bash 的 `\` 续行符；多行示例只出现在 docs 中，且会单独标注为 `bash`。

## 核心价值

你不需要先决定用哪个优化器、loss、neck、采样策略或论文方法。准备好模型和标注数据，用一句话告诉 YOLO Agent 当前问题；Agent 会自动建立 baseline、分析 COCO 错误、选择通过门禁的本地或论文 recipe、运行 matched pilot、用 ASHA 淘汰无效候选，并报告真实变化。

```powershell
# 当前模型误检多
yolo-agent train --model yolo26n.pt --data E:\dataset\coco.yaml --run-id reduce-fp --target-metric precision --target-delta 0.02 --goal-description "当前模型误检多，尤其是高置信度误检，请诊断并优化"

# 换了场景后效果不好
yolo-agent train --model yolo26n.pt --data E:\dataset\new-scene.yaml --run-id adapt-scene --target-metric map50_95 --target-delta 0.02 --goal-description "模型换到新场景后效果明显下降，请诊断场景偏移并优化"

# 小目标检测不好
yolo-agent train --model yolo26n.pt --data E:\dataset\coco.yaml --run-id improve-small --target-metric ap_small --target-delta 0.02 --goal-description "小目标漏检多，请提高 AP_small 并减少漏检"

# 提高整体 mAP
yolo-agent train --model yolo26n.pt --data E:\dataset\coco.yaml --run-id improve-map --goal +2map --goal-description "请提高整体 mAP，同时控制延迟和模型大小"
```

一句话描述负责指导诊断和 recipe 选择，指标与增量负责确定性验收；未显式指定目标时，可执行目标默认为 `+2map`。自动优化表示自动诊断并执行有预算、有证据的对照实验，不表示任何数据集或每次运行都保证提升 mAP。

- 一条命令完成环境检查、debug 训练、必要时的 mini-GPU 安全认证和有预算边界的 pilot 优化。
- 使用 matched baseline、本地 evidence、延迟和模型大小决定候选去留。
- 由 ASHA 管理 pilot 预算，尽早淘汰无效候选。
- Paper Intelligence 可离线导入论文目录并冻结为 `ResearchSnapshot`。
- 组件成熟度门禁阻止 metadata-only 或未验证 adapter 进入训练。
- 论文 recipe 必须绑定通过校验的 runtime adapter 和 matched control，ASHA 才会分配 pilot。
- 每个 run 都保存计划、事件、证据、队列状态和报告。

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

默认使用自动预算并只运行 pilot；full COCO 训练必须显式确认。修复环境后重跑同一条 `train` 命令即可。完整流程见[快速开始](docs/quickstart.md)，全部参数见 [CLI 与高级命令](docs/cli.md)。

## 决策流程（高层视图）

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

## 当前已验证状态（分四层）

状态拆成四个独立度量的层次。以下所有数字均来自 machine-readable artifact/config，不手工维护；日期与哈希以 artifact 为准。

### Research Coverage（研究覆盖）

论文目录很大不代表每篇都能用于 YOLO26；已实现 adapter、可执行 recipe、已复现论文是不同的度量。每轮自动优化会在 `artifacts/executable_portfolio.yaml`（首次自动轮次生成）报告完整搜索漏斗。详见 [Paper Intelligence](docs/paper-intelligence.md)。

<!-- paper-adapter-coverage:start -->
| 冻结论文记录 | 已实现 component IDs | 独立 Python adapter 类 | 源码声明 runtime components | Pilot reproduced components |
| --- | --- | --- | --- | --- |
| 728 | 158 | 103 | 0 | 0 |

这里的 158 是 component ID 数，实际由 103 个独立 Python adapter 类实现；两者都不是论文复现数量。artifact-backed 的本机 maturity 在下方验收表单独统计。
Audit snapshot: `c606d6c50fefaa7ae0db8bddb39d62057ff09ed5aeae943c81c990971b353e57`.

| Artifact 验收 | 结果 | 目标 |
| --- | --- | --- |
| 兼容论文有效 MethodProfile | 85/85 (100.0%) | >=85% |
| 兼容机制可复用 adapter | 20/23 (87.0%) | >=80% |
| 兼容机制 runtime integrated | 18/23 (78.3%) | >=70% |
| 兼容机制 smoke passed | 18/23 (78.3%) | >=60% |
| 兼容论文可复用 certified adapter | 83/85 (97.6%) | >=70% |

全目录 certified-adapter 映射覆盖为 83/728 (11.4%)；这是可复用组件适配，不是逐篇精确复现。

Exact reproduction 单独统计：0；separate detector family：168；insufficient information：475。
Acceptance hash: `797c3b912852717b03e3ce7fc55a3650d8b028f7d1dc9fc2a827c65c5996667c`.
<!-- paper-adapter-coverage:end -->

### Implementation Readiness（实现就绪，Paper-83 专项）

数据来源：`artifacts/pretraining_acceptance.yaml`（生成于 2026-09-22）；清单：`configs/research/paper_83_manifest.yaml`。

| 门禁 | 结果 |
| --- | --- |
| 专项清单论文数 | 83 |
| 规格完整 | 83/83 |
| 代码绑定实现 | 83/83 |
| Runtime 集成 | 83/83 |
| 单元测试 | 83/83 |
| 非 mock smoke 通过 | 83/83 |
| 兼容性校验 | 83/83 |
| Implementation ready | 83/83（blocked: 0） |

### Training Readiness（训练就绪）

数据来源：`artifacts/paper_83_runtime_preflight.yaml`（2026-09-21）、`artifacts/pretraining_acceptance.yaml`（2026-09-22）、`artifacts/training_release_v1.yaml`（2026-09-22）。

| 门禁 | 结果 |
| --- | --- |
| Runtime preflight | 83/83 通过；未知 runtime hooks：0 |
| Pretraining acceptance | 全部门禁 PASS |
| Training release | `READY_FOR_FIRST_TRAINING`，冻结于 git commit `3228dcc6` |

所有 artifact 中 `real_training_executed` 均为 `false`：这些门禁证明的是就绪状态，不是训练结果。

### Reproduction Evidence（复现证据）

数据来源：`docs/paper-adapter-coverage.yaml`（`pilot_reproduced_count`、`maturity_counts`）、`docs/paper-coverage-acceptance.yaml`（`exact_reproduction_paper_ids`）。

| 证据层级 | 数量 |
| --- | --- |
| Pilot reproduced 组件 | 0 |
| 精确复现论文（full） | 0 |
| 多种子确认 | 0 |

已实现不等于已复现。单次 pilot 提升只能标记为 `possible`，不能写成 `confirmed`；论文指标不能作为候选晋级证据。截至本快照，尚无任何本地复现证据。

## 能力边界

<!-- capability-maturity:start -->
| 能力 | 当前状态 | 代码存在 | 自动执行 | 本地复现 | 现实边界 |
| --- | --- | --- | --- | --- | --- |
| Pilot 自动训练 | `executable` | 是 | 是 | 取决于本地 run | 默认训练入口可执行 debug/pilot；是否成功取决于本机环境、数据和证据门禁。 |
| 自动导入基础指标 | `executable` | 是 | 是 | 取决于本地 run | 可导入 results.csv、训练 artifacts 和基础 runtime evidence；缺失产物仍会形成 evidence gap。 |
| Candidate COCO error facts | `incomplete` | 是 | 部分 | 部分 | 已有 post-eval、导入和 completeness gate；真实数据集仍需逐 run 验证完整 per-class/FN/FP/localization facts。 |
| Error-delta 下一轮决策 | `partial` | 是 | 部分 | 部分 | 能比较 parent/current error facts 并约束 proposal；证据不完整时只允许 evidence recovery。 |
| ASHA / successive halving 队列控制 | `executable` | 是 | 有门禁 | 未声明 | ASHA 是训练预算权威；full rung 仍必须显式确认，不能理解为默认自动跑完整 COCO。 |
| 论文组件 Adapter | `mixed` | 是 | 有门禁 | 未声明 | 已认证组件可经 MethodProfile、maturity、matched-control 和 ASHA 门禁进入 pilot；smoke passed 不等于 pilot reproduced。 |
| 3-seed confirmation | `executable` | 是 | 需显式确认 | 未声明 | 3-seed 确认工作流可端到端执行：pilot winner → 显式 full-run 审批（未审批不执行）→ 配对 3 seeds → paired 95% CI → 四档裁决（CONFIRMED/POSSIBLE/REJECTED/INCONCLUSIVE）→ confirmation_report；中断可 resume 且不重跑已完成 seed，审批单次消费。 |
| 稳定提升 +2 mAP | `not guaranteed` | 否 | 否 | 未声明 | +2 mAP 是优化目标，不是项目保证；必须由 matched baseline、full COCO、3 seeds 和置信区间证明。 |
<!-- capability-maturity:end -->

## 文档地图

- [系统架构（Canonical）](docs/architecture.md)
- [快速开始](docs/quickstart.md)
- [安装](docs/install.md)
- [CLI 与高级命令](docs/cli.md)
- [训练模式](docs/training-modes.md)
- [自动优化架构](docs/automatic-optimization-architecture.md)
- [COCO 与 YOLO26](docs/coco-yolo26.md)
- [自定义数据集](docs/custom-dataset.md)
- [数据优化与数据闭环](docs/data-optimization.md)
- [LLM 设置](docs/llm-setup.md)
- [证据模型](docs/evidence.md)
- [Paper Intelligence](docs/paper-intelligence.md)
- [论文 adapter 实现队列](docs/paper-adapter-implementation-queue.md)
- [论文 Recipe 执行门禁](docs/paper-recipe-materialization.md)
- [Distillation 机制](docs/distillation-mechanisms.md)
- [YOLO26 图结构组件](docs/yolo26-graph-components.md)
- [Awesome-object-detection 适配](docs/awesome-object-detection.md)
- [能力成熟度](docs/capability-maturity.md)
- [Paper-83 战役](docs/paper-83-campaign.md)
- [训练就绪流水线](docs/training-readiness.md)
- [训练发布](docs/training-release.md)
- [GPU Certification](docs/gpu-certification.md)
- [SAHI 推理认证](docs/sahi-inference-certification.md)
- [隔离推理策略 Adapter](docs/inference-policy-adapters.md)
- [故障排查](docs/troubleshooting.md)

## 开发

```powershell
python -m pip install -e ".[dev]"
pytest -q
ruff check .
```

默认 `pytest -q` 是快速回归套件，跳过大型 CPU/mock 集成测试；这些测试没有删除，需要完整验证时运行 `pytest -q --run-slow`。真实 CUDA 测试仍需额外使用 `--run-real-gpu` 显式开启。测试默认将小型 CPU 张量限制为单线程；需要调整时设置 `$env:YOLO_AGENT_TEST_TORCH_THREADS="4"`。

本项目采用 MIT License。
