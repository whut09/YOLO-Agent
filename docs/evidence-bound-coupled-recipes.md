# Evidence-bound Coupled Recipes

`EvidenceBoundCoupledRecipeLibrary` 只处理两个组件之间有明确互补机制的受控组合。它不是组合搜索器，也不会把一篇论文列出的全部组件自动打包。

## Allowlist

当前模板定义在 `configs/coupled_recipe_templates.yaml`：

| 组合 | 轨道 | 主要边界 |
| --- | --- | --- |
| P2 + small-object sampling | training | sampling 不归因 latency/model size |
| feature fusion + auxiliary quality loss | training | 不替换 native regression，不恢复 DFL |
| distillation + class-balanced sampling | training | teacher frozen，validation sampling 不变 |
| assignment + quality alignment | training | assignment shadow evidence 先通过 |
| slicing + confidence calibration | inference | checkpoint/training recipe 不变，标准 640 指标保留 |

模板只定义可接受的机制对，不能提供授权用的 `coupling_reason`。授权原因必须来自：

- `PaperMethodProfile` 中明确记录的 `coupling_reason` 和 source location；或
- 已验证的本地 diagnosis，且绑定当前 error fact IDs。

论文标题、同义 alias、同一论文内同时出现两个组件，均不能单独授权组合。请求必须恰好包含两个 canonical component，且 evidence 必须绑定同一组件对。

## 最小内部消融

每个 materialized recipe 固定生成四臂：

1. `baseline`
2. `A`
3. `B`
4. `A+B`

训练轨道由 `RecipeAblationPlanner` 生成 matched pilot 计划。A、B、A+B 都指向同协议 baseline control；所有四臂完成 post-eval 和 verified paired delta 前，不允许 ASHA 根据先验分数提前淘汰候选臂。最小 cohort 到达终态后，ASHA 才能按本地 paired evidence 裁剪后续预算。

Inference 组合由 `CoupledInferencePlanBuilder` 生成独立四臂计划。该计划不创建训练节点、不消费 ASHA 训练预算，并强制：

- `training_recipe=unchanged`
- `checkpoint=unchanged`
- `training_attribution_allowed=false`
- 标准 `imgsz=640` baseline 与 policy metrics 使用不同 namespace

## 贡献口径

`CoupledContributionAnalyzer` 只接受当前节点、未继承、已验证、协议匹配的 paired delta。同一 seed 的 A、B、A+B 必须使用同一个 matched control。

对每个允许归因的 metric 分别计算：

```text
A contribution           = delta(A)
B contribution           = delta(B)
combined total           = delta(A+B)
interaction contribution = delta(A+B) - delta(A) - delta(B)
```

arm 声明的 `attribution_excluded_metrics` 会作用于单组件和交互项。例如 sampling 可以报告 AP_small/FN 改变，但不能认领 P2 带来的 latency 或 model-size 改变。

单 seed 结果只能标记 `possible`。至少三个 matched seeds 且跨 seed 置信区间不跨 0，才可标记 `confirmed`；区间跨 0、缺臂、control/protocol 不匹配或 inherited evidence 都不能确认贡献。

## 失败方式

以下情况 fail closed：

- 组件对不在 allowlist。
- `coupling_reason` 不是显式 evidence。
- evidence 与请求组件不一致。
- 一次请求包含两个以上组件。
- inference-only recipe 尝试进入训练 RoundExecutionPlan。
- 任一 arm 缺少 matched control、post-eval 或 paired delta。

这些失败只产生拒绝或 evidence request，不会静默退化成普通 Ultralytics 训练。
