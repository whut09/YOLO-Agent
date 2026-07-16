# 能力成熟度矩阵

> 本页由 `configs/capability_maturity.yaml` 自动生成。请修改清单后运行以下命令：
> `python -m yolo_agent.tools.capability_matrix`，不要直接编辑表格。

最近审计日期：`2026-07-16`；Schema：`v1`。

这里刻意拆开三个概念：代码存在不代表可以自动执行，可以执行也不代表已经在本地复现；任何一项都不等于保证指标提升。

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

## 状态含义

- `executable`：已接入实际执行路径，但仍受环境、evidence 和 guard 约束。
- `incomplete`：存在主要模块，但关键证据链或协议仍不完整。
- `partial`：能覆盖一部分闭环，缺失条件下会降级或停止。
- `artifact_only`：目前只生成计划或 artifact，尚不权威控制执行。
- `mixed`：同一能力族包含不同成熟度的实现，必须逐项检查。
- `supported_not_automatic`：具备实现和门禁，但默认流程不会端到端自动完成。
- `not_guaranteed`：这是目标或期望结果，不是软件能力承诺。

## 源码依据

- **Pilot 自动训练**：`yolo_agent/agents/auto_optimization_loop.py`, `yolo_agent/adapters/ultralytics/training.py`
- **自动导入基础指标**：`yolo_agent/adapters/ultralytics/training.py`, `yolo_agent/core/coco_baseline_evidence.py`
- **Candidate COCO error facts**：`yolo_agent/adapters/ultralytics/coco_post_eval.py`, `yolo_agent/tools/coco_error_importer.py`, `yolo_agent/core/pilot_evidence.py`
- **Error-delta 下一轮决策**：`yolo_agent/agents/loop_evidence.py`, `yolo_agent/agents/policy_stage_runner.py`
- **ASHA / successive halving 队列控制**：`yolo_agent/agents/asha_scheduler.py`, `yolo_agent/core/round_execution_plan.py`, `yolo_agent/agents/auto_optimization_loop.py`
- **论文组件 Adapter**：`yolo_agent/components/contracts.py`, `yolo_agent/components/maturity.py`, `yolo_agent/components/adapters/registry.py`
- **3-seed confirmation**：`yolo_agent/agents/asha_scheduler.py`, `yolo_agent/agents/component_contribution.py`, `yolo_agent/agents/loop_policy_evaluator.py`
- **稳定提升 +2 mAP**：`yolo_agent/core/optimization_objective.py`, `yolo_agent/agents/component_contribution.py`
