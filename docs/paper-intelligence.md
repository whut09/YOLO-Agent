# Paper Intelligence

Paper Intelligence 是训练前的离线知识生产链。它把论文目录、摘要、note 和组件线索整理成可追溯的 ResearchSnapshot，供自动优化做诊断和候选筛选；它不把论文结论伪装成本地训练结果。

## 离线生产链

```text
catalog import
-> deduplicate
-> classify
-> component alias resolve
-> note and harness-hint parsing
-> contract and recipe prior generation
-> compatibility review
-> frozen ResearchSnapshot
```

训练开始后只读取冻结快照，不访问 live registry，也不联网读取论文。每个 base run 和 child run 都绑定同一个 `snapshot_hash`；快照变化会形成新的决策上下文，不能与旧快照下的 paper prior 混为一谈。

## Paper Claim 与本地 Evidence

论文标题、摘要、表格、reported delta、harness hints 和作者消融都只能标记为 `paper_claim` 或 `paper_prior`。它们可以帮助回答“值得研究什么”，但不能回答“本地候选是否提升”。

本地 promotion 只能依赖协议匹配的本地 evidence，例如 matched baseline、当前节点 COCO post-eval、paired delta、延迟、模型大小、paired bootstrap 和多种子置信区间。论文指标不能写入 candidate metric，也不能进入 local Pareto front。

论文库不是训练集。导入更多论文不会改变训练图片、标签或验证 split。

## Recipe Prior 与可执行 Recipe

论文方法首先生成不可执行的 RecipePrior。它必须绑定目标 error facts、组件 ID、来源位置、兼容性和预期改变变量，然后经过 materializer、eligibility gate、RecipeCritic、Utility/Budget 和 ASHA 才可能进入 pilot 队列。

- `metadata_only`：只有元数据，只能保留为研究记录。
- `recipe_idea_only`：只有配方想法，不是可执行 recipe。
- `adapter_required`：需要实现适配器，只能生成 implementation request。
- `adapter_implemented`：允许 dry-run，不代表可训练。
- `smoke_passed`：才允许进入受门禁的 pilot 候选。
- `pilot_reproduced`：已有本地 pilot 证据，不代表 full COCO 结论。
- `full_reproduced` / `confirmed_multi_seed`：需要显式 full 授权、匹配协议和多种子证据。

有论文记录不代表有 adapter；有 adapter 不代表 smoke passed；smoke passed 不代表 pilot reproduced；pilot reproduced 不代表 full COCO confirmed。

## 每轮决策边界

每轮只构建一次统一 DecisionContext，其中包含 baseline/current evidence、error facts、ResearchSnapshot、可执行 adapters、组件成熟度、兼容性、policy memory、已尝试动作、objective、预算和固定约束。

LLM 只能从输入提供的 paper/component IDs 中生成 doctor-style proposal。确定性的 RecipeCritic、eligibility、evidence、budget、ASHA 和 consent gate 拥有最终决定权。缺关键 evidence 时只能请求补证据；LLM 不能直接创建 `candidate_full`，也不能修改固定的 `imgsz=640`。

## 产物与可重放性

关键产物包括：

- `research_snapshot.yaml`：冻结论文、组件和 recipe 版本。
- `paper_recipe_plan.yaml`：本轮论文先验与候选计划。
- `component_compatibility.yaml`：兼容性和拒绝原因。
- `reproduction_state.yaml`：组件本地复现状态。
- `decision_ledger.jsonl`：规则/LLM 输入摘要、输出、critic 和 gate 结果。

空 catalog 或缺少快照时应明确报告 `paper_intelligence=unavailable`，并继续使用规则策略；系统不会假装引用论文经验。

导入和快照命令见 [Awesome-object-detection 适配](awesome-object-detection.md) 与 [CLI advanced 入口](cli.md)。
