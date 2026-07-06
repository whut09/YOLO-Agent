# 核心概念

YOLO Agent 把检测效果视为完整系统问题，而不仅是模型结构问题。

## 受控闭环

```text
任务 + 数据 + 错误样本 + 部署约束
        -> Diagnosis Graph 原因诊断
        -> 策略提案 + Utility Model 评分
        -> 受保护的候选实验
        -> 证据
        -> 下一轮
```

核心规则：LLM、人类和规则引擎只能提出策略；只有 evaluator 和 evidence gate 才能把策略变成实验候选。

Diagnosis Graph 会把 error facts 先映射成“症状、可能原因、需要补的证据、候选动作”。例如 `AP_small low` 不会直接等于“换 loss”，而会同时检查 feature stride、positive assignment、标注噪声、数据长尾和 slicing inference 等原因。

## State Machine And LLM Boundary

当前闭环是状态机驱动，不是自由聊天式 agent。`configs/loop_policy.yaml` 定义 stage 顺序、输入产物、输出产物、evidence gate 和 retry policy；`LoopState` 记录 completed/pending/blocked；`LoopOrchestrator` 只推进满足 contract 的 stage。

当前代码默认不调用大模型。大模型只被设计成可选的 `proposal_generator_only`：

- 可以生成：诊断摘要、policy proposals、需要补的 evidence、doctor report 草稿
- 不能生成：直接批准实验、直接启动训练、没有证据时声称最佳模型
- 必须经过：EvidenceGate、CompatibilityChecker、UtilityScorer、BudgetAllocator、StageContract、single-variable ablation guard

这种结构比“纯状态机”更灵活，也比“纯大模型决策”更稳：

```text
LLM / human / rules draft proposals
        -> state machine checks stage contract
        -> evidence and compatibility gates
        -> utility and budget scoring
        -> executable ExperimentNode / CommandSpec
```

大模型配置分两份：

- `configs/llm_decision.example.yaml`: 可提交脱敏配置，关键信息写 `XX`
- `configs/local/llm_decision.local.yaml`: 本地真实配置，Git 忽略，用于指定当前决策分析模型

Utility Model 会给每个 proposal 输出可解释分数，而不是只靠规则优先级：

```text
utility = expected_gain * confidence * target_error_relevance
          - training_cost - latency_risk - model_size_risk
          - implementation_risk - evidence_gap_penalty
```

因此候选进入实验前，会同时说明预期收益、置信度、目标错误相关性、训练成本、部署风险、实现风险和缺失证据。

## Policy Memory

Error Delta 不只用于生成下一轮建议，也会沉淀成长期策略记忆：

```text
action + target error fact + before/after delta + runtime cost + confidence
        -> runs/policy_memory.jsonl
```

例如某轮实验把 `AP_small` 从 `0.214` 提升到 `0.229`，且实际改动是 `loss.bbox.nwd`，系统会记录：

- action: `loss.bbox.nwd`
- target: `area_metric:small:ap_small`
- delta: `+0.015`
- cost: latency / model size 变化
- confidence: 单 seed 为 `low`，3 seeds 后才可能成为 `high`

如果没有 `changed_variables` 证明某个动作确实被执行，系统只会把 error fact 里的 action candidates 标记为 `inferred_action=true`，避免把“建议”误写成“因果”。未来同类任务遇到 small-object miss 时，Utility Model 可以查询历史 memory，而不是每次从零开始。

## Guarded Budget Optimization

Bandit / Bayesian Optimization 只用于“已通过 guard 的有限候选”，不能直接搜索组件空间：

```text
Diagnosis Graph / rules / human / LLM 提出 proposal
        -> compatibility + evidence gate + single-variable guard
        -> Budget Optimizer 在 accepted candidates 中分配预算
        -> Successive Halving 控制 pilot/full 晋级
```

默认 budget ladder：

- `pilot_3`: 先用小预算跑所有安全候选
- `pilot_10`: 只保留上一阶段 top candidates
- `candidate_full`: 只给最有希望且通过 promotion gate 的候选

这让系统从“遍历组件参数”升级为“在安全动作空间里做预算决策”：优化器只决定先跑谁、跑多少，不绕过证据门禁，也不直接批准 full COCO。

## Multi-Domain Actions

优化动作不是只有“换模型组件”。每个 error 都会尽量展开成同级候选：

- `model`: loss、head、assigner、neck 等组件变化
- `data`: hard negative mining、background-only injection、class rebalancing、small-object oversampling
- `augmentation`: mosaic/copy-paste/contrast/blur 等策略增减
- `postprocess`: threshold、Soft-NMS、SAHI、TTA 等推理策略
- `label`: missing-label check、box audit、class-definition review
- `training`: focal gamma、正样本分配压力、固定输入尺寸内的训练参数

这些候选都会进入同一个 `UtilityScorer`，按预期收益、目标错误相关性、证据置信度、训练成本、latency/model size 风险和实现风险排序。很多场景下，最优第一步会是补 hard negatives 或查漏标，而不是换 loss。

## Evidence-First Decisions

智能闭环不总是继续训练。证据不足时，系统会先生成 `evidence` 动作：

- `profile_data`: 补数据画像、类别分布、尺寸分布和数据健康度
- `advise_labels`: 补 label quality report、疑似漏标和错框复核
- `import_metrics`: 补 mAP、AP_small、per-class AP/AR、precision/recall
- `mine_errors`: 补 confusion matrix、false-positive samples、false-negative samples、localization errors
- `benchmark_latency`: 补 latency、FPS、model size

这些动作的 `execution_action` 不是 `run_training`，所以 queue/report 能明确区分“先补证据”和“跑候选训练”。没有关键 evidence 时，推荐应停在证据采集，而不是假装已经能选择最佳模型。

## Doctor-Style Decision Report

每轮 `next_round.yaml` 会写入 `doctor_report`。它不是候选列表，而是一次可审计的诊断：

- `primary_problem`: 当前优先问题，例如 `AP_small low`
- `likely_causes`: 可能原因，例如小目标低于有效 stride、采样不足、标注噪声
- `evidence`: 支撑证据，例如 `AP_small=0.21`、某类别 recall 偏低、错误样本统计
- `rejected_actions`: 被 guard 拒绝的动作和原因，例如固定 `imgsz=640` 时拒绝 `increase_imgsz`
- `selected_actions`: 本轮选中的动作，可能是训练、数据、标注、后处理或补证据
- `why`: 为什么这些动作直接针对当前错误事实
- `expected_improvement`: 只写可验证方向，例如 `AP_small increase; pilot_positive_delta required`
- `stop_condition`: 何时停止或不升 full，例如 pilot 没改善目标 error facts、latency 回退、证据仍缺失

因此优化策略不是遍历组件参数，而是：

```text
error facts -> causal diagnosis -> constrained actions -> utility scoring -> doctor report -> guarded execution
```

## 优化对象

- 模型尺寸和 YOLO family
- backbone、neck、head、loss、assigner、optimizer 元数据
- 标注质量和复标 worklist
- 数据健康度、采样、划分泄漏、重复帧
- 数据增强策略
- 后处理策略，例如 NMS、threshold、TTA、SAHI
- 部署限制，例如 latency、FPS、导出格式和模型大小
- 实验可复现性、消融纪律和证据质量

## 自动化成熟度

当前成熟度：Level 4，具备 Level 5 的基础模块。

- Level 1: schema + metadata
- Level 2: guarded candidate generation
- Level 3: evidence-driven loop
- Level 4: queued execution + cross-run learning
- Level 5: active learning + dataset version evolution

## 非目标

- 默认启动真实训练
- 复制未经验证的第三方 loss 实现
- 让 LLM 输出直接决定实验
- 在没有 evidence 时推荐最佳模型
- 用编造指标隐藏缺失 evidence
