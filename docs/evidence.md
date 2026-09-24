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

## 报告

```powershell
yolo-agent report --run runs/coco-yolo26n --out report.md
```

报告包含任务画像、数据诊断、候选模型、消融变量、指标表、最佳模型推荐、
下一轮建议、风险和未验证项。
