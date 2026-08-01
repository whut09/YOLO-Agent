# Paper Intelligence

Paper Intelligence 是训练前的离线知识生产链。它把论文目录、摘要、note 和组件线索整理成可追溯的 ResearchSnapshot，供自动优化做诊断和候选筛选；它不把论文结论伪装成本地训练结果。

## 离线生产链

```text
catalog import
-> deduplicate
-> classify
-> component alias resolve
-> note and harness-hint parsing
-> structured offline method evidence extraction
-> MethodProfile and adapter reuse decision
-> contract and recipe prior generation
-> compatibility review
-> frozen ResearchSnapshot
```

训练开始后只读取冻结快照，不访问 live paper registry 或 live maturity overlay，也不联网读取论文。每个 base run 和 child run 都绑定同一个 `snapshot_hash`、MethodProfile coverage 和有效 adapter identity；快照变化会形成新的决策上下文，不能与旧快照下的 paper prior 混为一谈。

## Paper Claim 与本地 Evidence

论文标题、摘要、表格、reported delta、harness hints 和作者消融都只能标记为 `paper_claim` 或 `paper_prior`。它们可以帮助回答“值得研究什么”，但不能回答“本地候选是否提升”。

本地 promotion 只能依赖协议匹配的本地 evidence，例如 matched baseline、当前节点 COCO post-eval、paired delta、延迟、模型大小、paired bootstrap 和多种子置信区间。论文指标不能写入 candidate metric，也不能进入 local Pareto front。

论文库不是训练集。导入更多论文不会改变训练图片、标签或验证 split。

## Recipe Prior 与可执行 Recipe

论文方法首先生成不可执行的 RecipePrior。它必须绑定目标 error facts、组件 ID、来源位置、兼容性和预期改变变量，然后经过 materializer、eligibility gate、RecipeCritic、Utility/Budget 和 ASHA 才可能进入 pilot 队列。

实际训练入口还会在 ASHA 注册时复验 runtime payload、plugin import、protocol hash、adapter patch hash 和 matched control。校验失败不会退化成普通 Ultralytics 训练；详细链路见 [Paper Recipe Materialization](paper-recipe-materialization.md)。

- `metadata_only`：只有元数据，只能保留为研究记录。
- `recipe_idea_only`：只有配方想法，不是可执行 recipe。
- `adapter_implemented`：已有可导入的 adapter 实现，但不代表运行时已接入。
- `runtime_integrated`：真实 entrypoint 和 runtime payload 已接入训练路径。
- `unit_tested`：对应 artifact 证明单元测试通过；mock smoke 仍不能提升后续状态。
- `smoke_passed`：非 mock smoke artifact 完整，才允许进入受门禁的 pilot 候选。
- `gpu_certified`：真实 GPU acceptance artifact 通过，不等于本地指标已提升。
- `pilot_reproduced`：已有 matched paired pilot evidence，不代表 full COCO 结论。
- `full_reproduced`：显式 full 授权下完成同协议 full reproduction。
- `confirmed_multi_seed`：至少三种子和置信区间确认，且所有 guard 通过。

`adapter_required` 是缺少实现时的门禁结果，不是可晋级的成熟度状态。成熟度只能按相邻状态推进，每次推进都必须绑定对应类型、文件哈希和通过状态的 artifact contract。GPU certification 失败会保留为 evidence，但不会提升 maturity。

`adapter_implemented` 之后的启动验证由 `ComponentValidationBridge` 在不创建训练节点的情况下完成；训练用的 `ComponentExecutionBridge` 不负责补成熟度，也不会接受 patch preview、mock smoke 或缺失/哈希失效的 smoke artifact。可恢复 artifact 约定见 [Paper Recipe Materialization](paper-recipe-materialization.md#runtime-maturity-bootstrap)。

机器相关的验证和认证状态写入独立的 [Component Maturity Registry](component-maturity-registry.md)，不会回写源码 component YAML。ResearchSnapshot 只冻结构建当时身份和 hash 均有效的 overlay；后续本地 registry 更新不会改变旧快照。

有论文记录不代表有 adapter；有 adapter 不代表 runtime integrated；runtime integrated 不代表 smoke passed；smoke passed 不代表 pilot reproduced；pilot reproduced 不代表 full COCO confirmed。

## 每轮决策边界

每轮只构建一次统一 DecisionContext，其中包含 baseline/current evidence、error facts、ResearchSnapshot、可执行 adapters、组件成熟度、兼容性、policy memory、已尝试动作、objective、预算和固定约束。

LLM 只能从输入提供的 paper/component IDs 中生成 doctor-style proposal。确定性的 RecipeCritic、eligibility、evidence、budget、ASHA 和 consent gate 拥有最终决定权。缺关键 evidence 时只能请求补证据；LLM 不能直接创建 `candidate_full`，也不能修改固定的 `imgsz=640`。

## 产物与可重放性

关键产物包括：

- `research_snapshot.yaml`：冻结论文、组件和 recipe 版本。
- `paper_recipe_plan.yaml`：本轮论文先验与候选计划。
- `component_compatibility.yaml`：兼容性和拒绝原因。
- `reproduction_state.yaml`：组件本地复现状态。
- `component_coverage_report.yaml`：论文提及、adapter 实现和 artifact-backed maturity 的分离计数。
- `paper_method_coverage.yaml`：每篇论文的方法 profile、adapter 复用决策和未实现原因。
- `paper_method_evidence.jsonl`：逐篇冻结 method family、canonical mechanism、插入点、改变变量、detector family、组件类型、训练/推理语义、兼容约束、runtime hooks、confidence 和 source location。
- `paper_method_evidence_coverage.yaml`：逐篇字段缺口、prior-only/authorizing 边界和相对上一冻结 snapshot 的 `insufficient_information` 迁移；Markdown 只保留摘要。
- `cached_code_metadata.yaml`：本地缓存 README/config 的文件 hash、来源路径和解析 warning；不包含网络抓取结果。
- `effective_component_maturity.yaml`：构建时有效的 adapter hash、Ultralytics 版本、认证 protocol 和 maturity artifacts。
- `decision_ledger.jsonl`：规则/LLM 输入摘要、输出、critic 和 gate 结果。

空 catalog 可以冻结为 `paper_intelligence=unavailable` 并继续使用规则策略。缺少快照、旧快照缺少 MethodProfile coverage，或 adapter/maturity 身份已变化时，真实 `train` 会在创建 run 前停止并要求重建 snapshot。

导入和快照命令见 [Awesome-object-detection 适配](awesome-object-detection.md) 与 [CLI advanced 入口](cli.md)。

尚未实现的论文组件由 [Paper Adapter Implementation Queue](paper-adapter-implementation-queue.md) 排入受诊断、本地 evidence、runtime hook、成本、去重和 cooldown 约束的工程队列。该队列只生成 implementation request，不自动生成未经验证的 adapter 代码。

同一 canonical mechanism 的论文映射、参数差异与实现复用规则见
[Paper MethodProfile 与 Adapter 复用](paper-method-profiles.md)。

## 离线 Method Evidence 边界

结构化提取只读取 catalog summary、本地 note、harness hints、title/year/category、official code metadata，以及用户显式提供的缓存 README/config。标题和 category 只能形成低置信度 prior；harness hint 只能形成诊断 prior。它们都不能单独授权 MethodProfile 实现或训练。

summary、note 或缓存代码元数据必须同时提供明确方法机制或 method family，以及 insertion point、changed variable、component type 或 required runtime hook，才会标记 `authorizes_method_profile=true`。这仍然不代表 canonical adapter 已实现，更不代表 `smoke_passed`。

早期工作区报告中的 `491` 来自旧 production artifact。Prompt 1 冻结 snapshot `e3b8d331...` 的可复现基线是 `480`；后续 delta 必须绑定 baseline snapshot hash，不能用构建中途被覆盖的 live production 文件计算。

本轮离线 evidence extraction 的验收 snapshot 为 `cc18ccb0922f...`，相同 catalog/commit 连续构建得到相同 hash。与 `e3b8d331...` 比较：`insufficient_information` 从 480 降为 445；41 篇因明确本地 method evidence 转为 `new_method_profile` 或 `new_component_adapter`，6 篇旧 title-only 假 adapter 映射被降级。`separate_detector_family` 保持 168。该迁移只表示论文方法边界更清楚，不表示新增 adapter、runtime integration 或本地复现。
