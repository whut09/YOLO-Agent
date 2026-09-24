# Evidence Architecture（证据架构）

YOLO Agent 的推荐必须基于 evidence。没有证据时，报告会写：

```text
No evidence, do not trust this result.
```

本文档定义证据体系的六个部分：层级（什么算证据）、身份（一条证据绑定了什么）、
错误剖面与配对差分（核心证据产物的真实字段）、完整性门（每种状态允许什么）、
决策追踪（决策如何被记录）、溯源（如何证明证据是真实产生的），以及声明边界
（每种证据可以说什么、不可以说什么）。

## Evidence 层级

证据按来源与可信度分层。**越往上（靠近论文）越不是本地事实；越往下（靠近
本地测量）越能授权训练与晋升。**

| 层级 | 含义 | 代码锚点 |
|------|------|----------|
| PaperClaim | 论文自述的声明，未经验证 | `PaperComponentClaim.evidence_level` 被钉死为 `"paper_claim"`（`research/schemas.py`）；`PaperBenchmark.verified` 默认 `False` |
| PaperPrior | 论文方法形成的配方先验，可参与规划、不授权实验 | `PaperRecord.evidence_level`、`RecipePrior`（`recipes/paper_priors.py`）；模块注释明确 paper claim "do not authorize an experiment or turn a paper claim into a trusted metric" |
| RuntimeEvidence | 组件运行时证据：runtime hook、单测、smoke | `ComponentMaturity` 十级阶梯（`components/maturity.py`）；`ComponentMaturityArtifact.artifact_sha256` + `mock` 标记——mock 证据不能支撑 `smoke_passed` 及以上 |
| DatasetEvidence | 数据集画像与清单 | `DatasetReport`（`tools/dataset_stats.py`）、`DatasetProfile`（`core/schemas.py`）、`dataset_manifest_sha256` |
| AnnotationEvidence | 标注质量与标注建议 | `label_quality_report`、`AnnotationAdviceReport`（`tools/annotation_advisor.py`）、label handoff 工件 |
| TrainingEvidence | 训练过程记录 | `runs/{run_id}/metrics.json`、run 协议版本（budget profile、epochs、seed、`batch_policy_hash`、`ultralytics_version`） |
| EvaluationEvidence | 固定协议的评估记录 | `MetricEvidence`（`core/experiment_graph.py`，含 `verified`、`protocol_hash`、`eval_protocol_hash`）；固定 COCO post-eval（`adapters/ultralytics/coco_post_eval.py`，`split="val"`、imgsz 锁 640） |
| DetectionErrorEvidence | 真实 GT + 预测导出的错误事实 | `DetectionErrorProfile`（`core/detection_error_profile.py`，builder 强制 "never from a mAP drop alone"） |
| PairedDeltaEvidence | 配对候选/父轮差分 | `DetectionErrorDelta`、`ErrorDeltaProfile`、`PairedExperimentResult`（`core/paired_experiment.py`） |
| DeploymentEvidence | 部署侧测量与硬约束 | `ResourceSnapshot`（latency/params/FLOPs/peak memory）、`TaskSpec.max_latency_ms` / `max_model_size_mb` 硬约束 |
| ConfirmationEvidence | 确认级证据：多种子统计 + 验收 + 同意 | ASHA 确认种子 95% 配对 Student-t CI、`PromotionRule` 裁决、`BaselineAcceptanceGate`、`FullRunConsent` |

代码中有三套并行的分级机制，不要混用：

1. `EvidenceLevel`（`research/schemas.py`）：`paper_claim → paper_prior →
   official_code_available → externally_reproduced → locally_smoke_tested →
   locally_pilot_reproduced → locally_full_reproduced → confirmed_multi_seed`。
   它是论文侧字面量，**代码中没有基于它的排序比较逻辑**。
2. `ComponentMaturity`（`components/maturity.py`）：真正的有序执行阶梯
   （IntEnum 0-9），升级只能 +1 且必须携带哈希绑定工件，降级需要
   `force=True` + reason。
3. `MetricEvidence.evidence_role`：run 记录级三值
   `current_observation | inherited_context | baseline_reference`，继承记录
   带 `origin_run_id` + `inheritance_depth`。

## Evidence Identity

每条 metric 记录（`MetricEvidence`，`core/experiment_graph.py`）绑定的真实
身份字段：

| 字段 | 说明 |
|------|------|
| `candidate_id` / `node_id` | 必填；记录属于哪个候选与实验节点 |
| `run_id` / `origin_run_id` | 产出 run；继承记录保留原始 run |
| `evidence_role` / `inheritance_depth` | 当前观测 / 继承上下文 / 基线参照 |
| `dataset_version` | 默认 `"unversioned"`——未升级清单的数据集是显式无版本，不是隐式可信 |
| `dataset_manifest_sha256` / `subset_manifest_sha256` | 数据清单哈希 |
| `split` | 默认 `"val"` |
| `protocol_hash` / `eval_protocol_hash` / `batch_policy_hash` | 训练协议、评估协议、batch 策略身份 |
| `seed` / `fidelity` / `epochs` / `imgsz` | 保真度身份 |
| `ultralytics_version` | 运行时版本 pin |
| `metric_name` / `value` / `higher_is_better` | 指标本体（方向自动推断） |
| `source` / `validator` | 默认 `"manual"`——手动导入的记录显式标记 |
| `verified` | 默认 `True`；未验证记录不参与晋升 |
| `metric_schema_version` | 当前 `"1.0"` |
| `created_at` | UTC 时间戳 |
| `source_artifact` / `confidence` | 来源工件与置信度 |

对照常见清单的诚实映射：

- **`experiment_id` 不存在**。记录身份由 upsert 六元组承担：
  `(origin_run_id or run_id, candidate_id, node_id, protocol_hash,
  evidence_role, metric_name)`（`core/evidence_store.py`）。
- **model/checkpoint hash 不在 MetricEvidence 上**。checkpoint 身份
  （`model_checkpoint_sha256` / `checkpoint_hash` / `base_model`）进入执行
  指纹 payload（`core/execution_fingerprint.py`）。
- **config hash 不是独立字段**。config 以 dict 存于 `Evidence.config` 并落盘
  `runs/{run_id}/config.yaml`；计划级哈希是 `ExperimentPlan.plan_hash()`，
  run 级身份是 `run_protocol_hash`。
- **producer 不在 MetricEvidence 上**（有 `validator`）。producer 概念由
  `ArtifactManifestEntry.producer_stage` 承担（见 Provenance 一节）。

`EvidenceStore` 的三条身份保障：

1. **原子 upsert**：同键记录按上述六元组整体替换（临时文件 + rename）。
2. **legacy 隔离**：来自旧协议 run 的 `current_observation` 记录在加载时被
   降级为 `inherited_context`（`inheritance_depth >= 1`，source 加 `legacy:`
   前缀）——旧数据永远不能冒充当前观测。
3. **manifest 校验**：只有 `entry.verify()` 通过的工件留在 evidence 映射；
   哈希失配的工件被剔除。

节点级指标落盘于 `runs/{run_id}/metrics_by_node.jsonl`，缺失项写入
`runs/{run_id}/artifacts/evidence_status.json`。

## DetectionErrorProfile

`core/detection_error_profile.py`，schema `detection_error_profile.v1`。由
真实 GT 与真实预测构建；profile 级身份字段：`profile_id`（必填）、`run_id`
（必填）、`candidate_id`（必填）、`node_id`、`split`
（`val|test|train`）、`role`（`baseline|candidate`）、`protocol_hash`、
`gt_artifact`、`predictions_artifact`、`dataset_manifest_hash`。

| 分节 | 真实字段 |
|------|----------|
| `global` | `map50`、`map50_95`、`precision`、`recall` |
| `scale`（COCO area 桶） | `ap_small` / `ap_medium` / `ap_large`、`recall_small` / `recall_medium` / `recall_large` |
| `per_class` | 每类：`category_id`、`name`、`ap`、`ap50`、`precision`、`recall`、`support` |
| `false_negative` | `total`、`by_class`、`by_scale`、`by_confidence`、`by_scene_slice`（小目标 FN 走 `by_scale` 键，无独立字段） |
| `false_positive` | `total`、`by_class`、`background_fp`、`duplicate_fp`（重复检测）、`class_confusion_fp`、`high_confidence_fp` |
| `localization` | `matched_iou_distribution`（IoU 桶分布）、`mean_matched_iou`、`localization_error_count`、`ap50_vs_ap75_gap`（无 center error 字段） |
| `classification` | `confusion_matrix`、`top_confusion_pairs` |
| `confidence` | `tp_confidence_histogram`、`fp_confidence_histogram`、`calibration_bins`、`expected_calibration_error`（ECE；无 mean_confidence 字段） |
| `scene_slices` | 每片：`slice_name`、`gt_count`、`true_positives`、`recall`、`ap50`；数据集未提供 slice 元数据时为空，并记录 `unavailable_scene_slices`——**绝不伪造** |
| `resources` | `ResourceSnapshot`：`latency_ms`、`params`、`flops_g`、`peak_memory_mb`——只来自 run 自身的 resource manifest（latency audit / FLOPs audit / peak memory），**绝不由 agent 估计**；无快照时该节为 `None` |

## DetectionErrorDelta

`core/detection_error_delta.py`，schema `detection_error_delta.v1`。输入是
**两个** matched 的 `DetectionErrorProfile`（parent/baseline 与
candidate），输出逐维差分。

**matched 评估门**：仅当两侧 `gt_artifact`、`dataset_manifest_hash`、
`split` 完全一致才记 `matched_evaluation=True`；不一致时差分仍会计算，但
`matched_evaluation` 被显式标记为 `False`——消费方（ASHA 配对有效性检查）
会拒绝未匹配的配对证据。

逐维比较（section 顺序即生成顺序）：

| Section | 比较内容 | 方向 |
|---------|----------|------|
| `global` | `map50`、`map50_95`、`precision`、`recall`（accuracy 维度） | 越高越好 |
| `scale` | `ap_small` / `ap_medium` / `ap_large`（recall_* 不进差分） | 越高越好 |
| `per_class` | 按类名配对；`{name}.{ap\|precision\|recall}`（parent 没有的类跳过） | 越高越好 |
| `false_negative` | counts：`total`、`by_scale.{key}`、`by_class.{key}`（`by_confidence`/`by_scene_slice` 不进差分） | 计数越低越好 |
| `false_positive` | counts：`total`、`background_fp`、`duplicate_fp`、`class_confusion_fp`、`high_confidence_fp` | 计数越低越好 |
| `localization` | count：`localization_error_count`；metric：`ap50_vs_ap75_gap` | 计数越低越好；gap 越小越好 |
| `confidence` | metric：`expected_calibration_error` | 越小越好 |
| `classification` | 仅当任一侧有混淆矩阵：counts `confusion.{key}`、`top_pair.{label}` | 计数越低越好 |
| `resources` | **仅当两侧都有 ResourceSnapshot**：metrics `latency_ms`、`flops_g`、`peak_memory_mb`（latency/memory 维度），count `params`；单侧缺值即跳过该项，绝不造数 | 越低越好 |

裁决语义：每个 delta 是 `MetricDelta`（float：`candidate`、`parent`、
`delta`、`verdict`、`higher_is_better`）或 `CountDelta`（int）。verdict ∈
`improved | regressed | unchanged | not_comparable`；浮点容差仅
`epsilon=1e-9`，值 round 到 6 位小数。**真正的噪声门不在这里**：诊断晋升
与 candidate promotion 用 `0.0005` 噪声下限（`agents/diagnosis_promotion.py`
等），ASHA `pilot_3` 用 `-0.0015`（`agents/asha_scheduler.py`）。

同名注意：`agents/error_delta_profile.py` 另有一个同名 `MetricDelta`，字段
完全不同（`metric_name`、`scope`、`subject`、`baseline_value`、
`candidate_value`、`delta`、`relative_delta`、`higher_is_better`、
`improvement`、`seed_values_*`），服务于多指标 profile 级差分
（`ErrorDeltaProfile`）。它还定义：

- `EvidenceGap`：缺失指标记为
  `missing_in_candidate / missing_in_baseline / missing_entirely`——**缺失
  即显式 gap，绝不造数**；
- `HardConstraintViolation`：部署硬约束（`max_latency_ms` 对
  `latency_ms`、`max_model_size_mb` 对 `model_size_mb`）超限即记录，limit、
  observed、baseline_observed 三值齐全；
- 必需指标集 `REQUIRED_DELTA_METRICS`（`map50_95`、`precision`、`recall`、
  `ap_small`、`ap_medium`、`ap_large`、`latency_ms`）——"证据不完整必须
  可见" 的最低要求。

## Evidence Completeness

完整性的真实代码状态（无大写常量，全部小写/布尔）：

**轮级三态门**（`EvidenceVerdict`，`core/error_profile_evidence_gate.py`，
schema `error_profile_evidence_gate.v1`）——检查 8 个必需分节（global、
scale、per_class、false_negative、false_positive、localization、
classification、confidence）：

| verdict | 判定 | 允许什么 |
|---------|------|----------|
| `complete` | 8 节齐全且 matched | `next_round_experiment`；**唯一允许请求下一轮训练预算的状态**（`budget_request_allowed=True`） |
| `partial` | 有缺口但不严重 | 仅证据修复动作：`collect_evidence`、`rerun_eval`、`repair_dataset_metadata`、`repair_evaluation`；预算被拒（`budget_request_blocked_until_evidence_complete`） |
| `insufficient` | 缺 profile、缺节 > 4、或 delta 不 matched | 仅 `collect_evidence`（`budget_request_blocked_evidence_insufficient`） |

该门**从不伪造证据，也不阻断诊断与 pilot**——它只管"下一训练轮"的预算
请求。gate 不放预算时，决策追踪里选择预算型实验会直接 raise。

**节点级布尔门**（`PilotEvidenceCompletenessGate`，`core/pilot_evidence.py`）
——检查 7 个必需指标 + per_class 前缀 + 3 个工件（coco_predictions /
coco_eval / coco_error_report，SHA256 校验 + contract hash）+ 4 个报告字段
+ 2 个 fact 组，且只统计 verified、同 run/candidate/node/protocol、
depth=0 的记录：

- `complete=True`：训练提案照常。
- `complete=False`：产出 `evidence_actions` 修复清单
  （`run_coco_post_eval` / `import_coco_eval` / `mine_coco_errors` /
  `repair_coco_evidence_artifacts` / `import_current_node_error_facts`），
  写入 `artifacts/pilot_evidence_completeness.yaml`，发 `contract_blocked`
  事件，并把 `next_round.yaml` 改写为 `proposal_mode="evidence_only"`、
  `training_proposals_allowed=False`——下一轮只许收证据，不许跑训练。

**ASHA 侧**（`agents/asha_scheduler.py`）：观测带 `evidence_complete`
布尔；`report()` 遇到 `evidence_complete=False`、配对 delta 缺失或配对
未验证 → trial 置 `needs_evidence`，该分支优先于一切晋级分支。
`needs_evidence` 的 trial 拿不到新 assignment、不参与晋级统计；晋升要求
完整证据 + 验证过的配对 + 诊断门通过。预注册 trial 的初始状态就是
`needs_evidence`。

**晋升侧**（`agents/promotion_rule.py`）：证据矩阵不完整、主指标缺失、
约束证据缺失，一律 `INCONCLUSIVE`（fail-closed）。

一句话总结：**planning 永远允许（门不挡诊断）；training 只在 complete 时
允许；promotion 在任何不完整证据下一律拒绝。**

## Decision Trace

`core/error_decision_trace.py`，schema `error_decision_trace.v1`，落盘
`decision_trace.yaml`。一轮决策的完整审计字段：

| 字段 | 内容 |
|------|------|
| `status` | `pending_observation → observed`，或 `blocked_by_evidence` |
| `problem` | 由真实数字生成的问题陈述（如 "open error budget: N FNs and M FPs on split 'val'"），不是自由感慨 |
| `evidence_profile_id` / `evidence_delta_id` | 本轮决策依赖的证据 ID |
| `gate_verdict` | 完整性门的 `complete / partial / insufficient` |
| `hypotheses` | 证据链接的假设（`EvidenceLinkedHypothesis`） |
| `candidate_actions` | 候选动作清单 |
| `rejected_actions` | `{family, reason}`——拒绝原因必填非空 |
| `selected` | `{family, expected_effect, round_budget_requested, gate_allows_budget}`：期望效果是显式记录（自由文本），预算请求与门放行成对出现 |
| `observed` | `{delta_source, improvements, regressions}`——只从真实 candidate-vs-parent delta 填充 |
| `next_decision` | 仅 `observed` 状态可携带（校验强制） |

不变量：`observed` 状态必须有 `observed`；`blocked_by_evidence` 不得有
`observed`；预算被 gate 扣留时请求预算会 raise。运行时闭环由
`close_round`（`core/error_trace_next.py`）完成：记录观测 →
`decide_next_round` → 写回 `next_decision` → 落盘。能力晋升
（`error_capability_promotion.py`）要求 trace 的证据 ID 与 delta/profile
匹配、候选动作与观测非空才放行。

## Provenance

证据如何证明"是谁、用什么、在哪份代码下产生的"：

- **Artifact manifest**（`core/artifact_manifest.py`，append-only JSONL，
  `runs/{run_id}/artifacts/artifact_manifest.jsonl`）：每条工件记录
  `name`、`type`（file/directory）、`path`、`sha256`（文件取文件哈希，目录
  取稳定树哈希）、`producer_stage`、`run_id`、`candidate_id`、`node_id`、
  `protocol_hash`、`created_at`、`schema_version`。`verify()` 现场重算哈希
  比对。
- **执行指纹**（`core/execution_fingerprint.py`，schema
  `execution_fingerprint.v1`）：payload 含 checkpoint 身份、规范组件 ID、
  recipe id/version、有效 overrides（imgsz/seed 被排除在外）、数据集
  manifest 哈希、baseline 协议哈希、imgsz、fidelity、seed、teacher
  checkpoint 哈希、图身份哈希、runtime payload 哈希、组合指纹——规范化
  JSON 的 sha256。paper ID 被刻意排除。`paired_evidence_is_valid` 要求配对
  verified + 协议 matched + matched control matched + imgsz=640 + 协议/
  数据集/保真度哈希逐一相等。
- **代码身份在 run 级**：`RunProtocolVersion`（`core/run_protocol.py`）含
  `code_version = current_code_version()`（git HEAD + dirty 标记，回退包
  版本）、`ultralytics_version`（importlib 实测）、
  `dataset_manifest_sha256`、`batch_policy_hash`、`eval_protocol_hash`，串成
  `run_protocol_hash` 写入 run context。`MetricEvidence` 本身没有
  code_version 字段——代码身份由 run 级协议哈希承载，这是刻意的分层。
- **Runtime source identity**：研究快照冻结
  （`research/production_pipeline.py` + `snapshot.py`）——每个工件 pin
  sha256，快照目录以哈希命名，`ResearchRuntimeBinding` 把
  `research_snapshot_hash` 绑定到训练 run，**绝不回退到 live registry**；
  不可用时写显式常量 `UNAVAILABLE_RESEARCH_SNAPSHOT_HASH`。adapter 侧
  `adapter_runtime_protocol_hash` + `adapter_runtime_payload_hash` 在
  runtime entrypoint 与 execution bridge 处校验失配即拒。
- **Release identity**（`research/training_release.py`）：`TrainingRelease`
  自哈希（`release_hash`）+ pin 全量证据哈希（manifest membership、验收、
  注册表、preflight、non-GPU、测试结果、组件 runtime/identity 哈希、
  adapter 源码哈希 + runtime 依赖哈希、`git_commit`）。
  `verify_training_release` 逐项现场复查（全部 fail-closed），pin 的
  commit 必须等于当前 HEAD 或其祖先；不通过即抛
  `TrainingReleaseMissingError`——**每个真实训练入口必须先过此关**。

## Claims：每种证据可以说什么

| 证据级别 | 可以说 | 不可以说 |
|----------|--------|----------|
| smoke（smoke_passed） | "runtime works"——组件在该协议下能跑通 | 任何效果声明、任何精度含义 |
| pilot（pilot_3 / pilot_10） | "possible"——筛选信号，可排序、可晋升入下一档 | "确认有效"、"复现了论文" |
| full（candidate_full 单 run） | "full-run observed"——完整预算下观测到该结果 | "已确认"——单 run 无统计效力 |
| 3-seed CI（confirmation seeds 42/43/44） | "confirmed under this protocol"——配对 95% Student-t CI 下界 > 0 | "普遍有效"——结论限于该协议、该数据集、该 baseline |

越级声明在代码上被四处强制：

1. `PaperComponentClaim` 钉死 `paper_claim`，论文声明永不授权实验；
2. 成熟度只能 +1 逐级过渡且必须携带哈希绑定工件，mock 证据到不了
   `smoke_passed`；
3. `needs_evidence` / 证据不完整的 trial 与矩阵永远拿不到晋升
   （ASHA report 分支优先 + `PromotionRule` fail-closed `INCONCLUSIVE`）；
4. `MetricEvidence.verified=False` 或配对无效的记录不参与任何统计。

推理策略指标（`sliced_*` / `tta_*` / `calibrated_*` 等前缀）属于独立 metric
namespace：永远不进入 `standard_640` 基线，也不参与训练组件归因。因此它们不能
作为任何训练侧结论的证据，只能支撑部署侧的精度/延迟权衡声明。边界定义见
[Isolated Inference Policy Adapters](inference-policy-adapters.md)。

## 报告

```powershell
yolo-agent report --run runs/coco-yolo26n --out report.md
```

报告包含任务画像、数据诊断、候选模型、消融变量、指标表、最佳模型推荐、
下一轮建议、风险和未验证项。
