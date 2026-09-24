# Documentation Acceptance — 最终文档验收报告

- **验收日期**: 2026-09-24
- **验收范围**: 全部面向用户的文档（README×2 + `docs/` 61 个 markdown）+ 支撑文件（`docs/*.yaml`、`docs/*.svg`）
- **验收方法**: 代码为唯一事实源。四路逐项对照（每子项给出文档侧与代码/artifacts 侧 file:line 证据）+ 11 个禁止冲突关键词逐词上下文审查 + 全量门禁（docs consistency tests、`pytest -q`、`pytest -q --run-slow`、`ruff check .`）。
- **性质**: 验收轮。不新增功能；发现的文档精度错误当场修复（共 9 处，见第 4 节），修复后重跑门禁。

---

## 1. 验收总览

| 指标 | 结果 |
|---|---|
| Total docs | 63 个 markdown（`docs/` 顶层 59 + `docs/generated/` 2 + README×2），另有 3 个 yaml/svg 支撑文件 |
| Broken links | **0**（`test_docs_links` 覆盖 README×2 + `docs/**/*.md` 的全部相对链接，含图片与锚点剥离） |
| CLI example failures | **0 / 140**（文档中全部 140 条 `yolo-agent` 命令逐条通过 `build_parser().parse_args()`；`setup/train/status/stop` 四个子命令确认在文档中出现） |
| Stale generated blocks | **0**（`docs_status --check` 报告 current；`capability_matrix --check` 报告 current；`architecture_sources --check` 报告 fresh） |
| Bilingual drift | **0**（EN/ZH 标题 20 对一一对应；docs/ 链接各 31 条且集合相等；status 块由同一结构化源渲染；8 条命令参数骨架逐字一致，仅 `--goal-description` 按语言本地化） |
| Known capability mismatches | 8 条 capability 中 4 条为 manifest 显式声明的非可执行状态（`incomplete` 1、`partial` 1、`mixed` 1、`not_guaranteed` 1），全部与代码现状相符，见第 5 节 |
| Architecture coverage | 8 个 area（cli/task/evidence/action_space/runtime/experiment/research/release）全部纳入 `docs/ARCHITECTURE_SOURCES.yaml` 快照追踪，当前 fresh；`docs/architecture.md` 抽查 30+ 概念→包映射全部真实存在 |

---

## 2. 十领域验证矩阵

每项结论为 **OK**（与代码一致）、**OK [declared]**（一致且含诚实标注的限制）或 **FIXED**（验收中发现精度错误并已修复）。

### 2.1 README EN/ZH（5/5 OK）

| 子项 | 结论 | 证据 |
|---|---|---|
| product description | OK | 20 对标题一一对应（"Core value"/"核心价值"等），四个问题导向小节双侧同构 |
| commands | OK | 8 条命令参数骨架逐字一致（`README.md:19-28,50` ↔ `README.zh-CN.md:19-28,50`）；差异仅为 `--goal-description` 自然语言本地化 |
| status | OK | 两侧 status 块均由 `python -m yolo_agent.tools.docs_status` 从同一 7 源结构化数据渲染（`test_readme_bilingual_status_sync` 强制） |
| capability | OK | `paper-adapter-coverage` 与 `capability-maturity` marker 块由生成器管理，双侧各恰一对（测试强制） |
| links | OK | docs/ 链接 31 = 31 且集合相等；全部链接可解析 |

### 2.2 Architecture（5/5 OK，2 处 FIXED）

| 子项 | 结论 | 证据 |
|---|---|---|
| system map | OK | 14 planes 的 75+ 模块路径抽查全部真实存在；runtime entrypoint 进程启动重查 gate（`adapters/ultralytics/runtime_entrypoint.py:63-69`）；Mermaid 节点全部指向真实类 |
| authority boundary | OK | LLM 禁改 imgsz/禁请求 candidate_full（`agents/llm_decision_advisor.py:205-209`）；ASHA 隔离未授权 trial（`agents/asha_scheduler.py:348-350,211`）；full-run 三条件（`cli.py:1197`、`core/full_run_consent.py:35-38`）；manifest frozen + membership_hash（`configs/research/paper_83_manifest.yaml:4,18`） |
| state machine | OK [FIXED×1] | 50+ 枚举值逐字比对一致（LoopStage 15 值、StageStatus 6、QueueStatus 9、ASHATrialStatus 9、PromotionVerdict 4、dispositions 7）；supersede 命名格式（`auto_optimization_loop.py:5283,5320`）；**FIXED**：round cap 的配置键名说明修正（原暗示 YAML 键为 `max_rounds_safety`，实际 YAML 键为 `max_auto_rounds_safety`，Python 字段名不变，`core/optimization_budget.py:24,58`） |
| artifact flow | OK | 10 行概念→类映射全部核实（`DetectionErrorProfile`/`ActionSpec`/`ExperimentNode`/`MetricEvidence`/`DetectionErrorDelta`/`ErrorDecisionTrace`/`PolicyMemoryRecord` 等）；四条函数级连接逐一验证（`data_stage_runner.py:43,99`、`loop_policy_evaluator.py:485,768`、`execution_queue.py:52`、`detection_error_delta.py:142-146`） |
| package map | OK [FIXED×1] | 30+ 映射抽查全部存在；五个大模块行数引用吻合（8.5k/7.6k/2.2k/1.75k/1.7k）；**FIXED**：包行数 "roughly 75k" 已过时，实测 `yolo_agent` 包 167,106 行（`agents` 40.1k + `research` 32.3k + `components` 31.1k + `core` 22.5k + `certification` 16.0k + 其余），已更新 |

### 2.3 Evidence（5/5 OK）

| 子项 | 结论 | 证据 |
|---|---|---|
| profile | OK | schema `detection_error_profile.v1`；10 个身份字段 + 10 个分节字段逐一对照（`core/detection_error_profile.py:161-203`）；文档的"无 center error / 无 mean_confidence"负向断言与代码一致 |
| delta | OK | 9 个 section 顺序与语义逐一吻合（`core/detection_error_delta.py:150-358`）；噪声门 0.0005（`agents/diagnosis_promotion.py:23-24`）与 ASHA pilot_3 的 -0.0015（`asha_scheduler.py:1263`）；两个同名 `MetricDelta` 的辨析逐字段一致 |
| trace | OK | schema `error_decision_trace.v1`；problem 格式串逐字吻合（`core/error_decision_trace.py:130-134`）；三条不变量与 `close_round` 闭环（`core/error_trace_next.py:59`） |
| provenance | OK | 五层溯源全部核实：artifact manifest producer_stage + verify 现场重算（`core/artifact_manifest.py:25,72`）、执行指纹三键 + paper ID 刻意排除（`core/execution_fingerprint.py:87-89,5,174`）、RunProtocolVersion、ResearchRuntimeBinding + UNAVAILABLE 常量、release_hash 自哈希（`research/training_release.py:155-157`） |
| claim maturity | OK | 三套分级机制分离（`EvidenceLevel` 八级无排序逻辑、`ComponentMaturity` IntEnum 0-9 升降级约束、`MetricEvidence.evidence_role`）；needs_evidence 优先于晋级分支（`asha_scheduler.py:612-620`）；INCONCLUSIVE fail-closed（`agents/promotion_rule.py:18-19`） |

### 2.4 Research（5/5 OK，1 处 FIXED 在 coverage 汇总页）

| 子项 | 结论 | 证据 |
|---|---|---|
| paper catalog | OK | `paper_count: 728`（`docs/paper-adapter-coverage.yaml:6` ↔ `tools/paper_adapter_coverage.py:34`）；baseline 分母漂移 84 已在 campaign 文档显式声明（`paper-83-campaign.md:30-33` ↔ `research/production/coverage_baseline.yaml:744`） |
| MethodProfile | OK | 互斥字段校验（`research/method_profiles.py:164-169`）；六类 ImplementationDecisionKind 逐字一致（`:44-50`）；mechanism 三计数与比率字段存在于 `paper_method_coverage.yaml` |
| component adaptation | OK | 158/103/0 三计数与 maturity_counts 生成逻辑一致（`tools/paper_adapter_coverage.py:50-53,144-147`） |
| Paper-83 campaign | OK [declared] | `Literal[83]` 双重 pin（`research/paper_83_campaign_schemas.py:72,103`）；membership_hash 校验（`:107-116`）；**declared**：85 分母仅存在于冻结历史报告（`docs/paper-coverage-acceptance.yaml:341`，文档已标注 historical） |
| reproduction | OK [declared] [FIXED] | `pilot_reproduced_count: 0`、`exact_reproduction_paper_ids: []`、runtime_integrated 及以上全 0；**FIXED**：`docs/coverage-baseline-summary.md` 原引用旧快照数字（80/80/47/2/13 classes），已重写为当前生产 baseline（84/84/0、reusable 86、runtime-ready 84）并以 `report_hash 882d0e25…` 锚定来源 |

### 2.5 Runtime（4/4 OK）

| 子项 | 结论 | 证据 |
|---|---|---|
| component | OK | `ComponentEvidenceOverlay` 五类字段与 registry 实际结构逐项对应（`runs/component_maturity_registry.yaml:2-13`，637 overlays）；10 级 ladder 与"只推进相邻状态"（`components/maturity.py:33-55`、`maturity_certification.py:56-66`） |
| materialization | OK | RecipeCritic → RoundExecutionPlan → ExecutionQueue → ASHA → materialization gate 全链有实现（`agents/recipe_critic.py:58`、`core/execution_queue.py:151`、`agents/paper_recipe_materialization/gate.py`）；三个 content-hash-bound artifact 命名与 registry 实际文件吻合 |
| preflight | OK | `paper_count/passed/failed/unknown_runtime_hooks` 六字段 schema 逐字一致（`research/paper_runtime_preflight.py:108-119` ↔ `artifacts/paper_83_runtime_preflight.yaml:2-8`）；`RuntimeHookIdentity` 字段表在 artifact record 逐项可见 |
| GPU cert | OK [declared] | 四层命令面全部存在（`cli.py:7291,7243,7077-7085,1155-1589`）；`gpu_certified ≠ paper reproduced` 由 `certification/paper_auto_optimization_maturity.py:151-155` 强制；mini-GPU artifact 真实存在（`runs/certification/mini-gpu/certification_report.yaml`）；**declared**：不证明 +0.02 mAP50-95、不授权 full COCO |

### 2.6 Governance（4/4 OK）

| 子项 | 结论 | 证据 |
|---|---|---|
| pretraining acceptance | OK | 七 section + 9 个 IntegrityCheck 与 artifact 逐项对应（`research/pretraining_acceptance.py:625-682` ↔ `artifacts/pretraining_acceptance.yaml`）；`training_gate.allowed ≡ verdict=="PASS"` 不变量（`:369-389`）；`real_training_executed: Literal[False]`（`:343`） |
| training gate | OK | 12 个 `GateLockReason`（`research/paper_83_training_gate.py:49-62`）；四道 seam（L0 `cli.py:2352-2365`、optimize `cli.py:2682-2689`、executor `core/executor.py:117,229,331`、runtime entrypoint exit 86 `adapters/ultralytics/runtime_entrypoint.py:60-73,201-209`） |
| training release | OK | `release_status` 仅两值（`research/training_release.py:34`）；git 冻结 HEAD-or-ancestor + 不可用 fail-closed（`:201-223,716-737`）；drift → exit 2 提示语逐字一致（`cli.py:2366-2390` ↔ `docs/training-modes.md:150,166`）；release_id 格式与 artifact 前缀精确匹配 |
| full-run consent | OK | `FULL_RUN_CONFIRMATION_PROFILES` 三值逐字一致（`agents/optimize_runner.py:63`）；consent 绑定四 hash + GPU-hours cap 24（`core/full_run_consent.py:35-38,197-203`、`configs/training/yolo26_coco_goal.yaml:12`）；consent 文件可代确认（`optimize_runner.py:235`） |

### 2.7 Training（4/4 OK，1 处 FIXED）

| 子项 | 结论 | 证据 |
|---|---|---|
| dry/debug/pilot/full | OK | 五 profile 数值（fraction/epochs/seeds/val/quick_val）YAML ↔ 代码默认逐项一致（`configs/training/yolo26_coco_goal.yaml:111-176` ↔ `adapters/ultralytics/training.py:83-140`）；dry-run 不设门语义（`cli.py:1192-1195,2357-2358`） |
| auto advance | OK | `_next_auto_profile` 链定义（`optimize_runner.py:1234-1241`）；`--no-auto-advance`/`--auto-rounds 0` 语义（`cli.py:1211-1215,1183-1190`）；full 候选写 `full_candidate_recommendations.yaml`（`auto_optimization_loop.py:1287`） |
| resume | OK [FIXED×1] | train 无 `--resume`、profile 从 run_context.yaml 推断（`cli.py:2447-2462`）、确认矩阵保留已完成 seed（`optimize_runner.py:1350`）；**FIXED**：`docs/cli.md:33` 收紧为"存在未完成工作的同名 run 会自动恢复，已完成或停滞的同名 run 得到递增后缀"（`core/run_allocation.py:94-108`） |
| 3-seed | OK | 确认 seeds `[42,43,44]`（`asha_scheduler.py:192`）；四 rung 数值逐一一致（`:1253-1287`）；7 个淘汰原因字面量逐字存在（`:881-1004`）；paired CI 下界 > 0（`:997`）+ 95% Student-t（`:1346-1355`） |

### 2.8 Inference（5/5 OK，1 处 FIXED）

| 子项 | 结论 | 证据 |
|---|---|---|
| postprocess | OK [FIXED×1] | INFERENCE_ONLY_FAMILIES 四族（`agents/action_space_schemas.py:64-71`）；策略目录 18 条含三 NMS 变体（`configs/postprocess_strategies.yaml`）；**FIXED**：family 计数 26 → 27（`active_learning` 族加入后文档未同步，实测 `ActionFamily` Literal 27 值） |
| TTA | OK | kind → `tta_inference` namespace（`components/adapters/inference/policy.py:15-41`）；latency 警告与两种计时协议不可比的声明与代码一致（`backend.py:50-87` 无 warmup ↔ `adapters/ultralytics/inference_latency.py:31-34` 3 warmup + 20 timed） |
| SAHI | OK | `certify-inference-policy` 的 `sahi_slicing` 走单视图 640 后端、`certify-sahi` 才是真切片（`backend.py:53-72` ↔ `cli.py:7418-7450`）；缺依赖产出结构化 `skipped`（`sahi_runner.py:148-151`） |
| calibration | OK | temperature 非中性校验（`policy.py:75-76`）；后端 `calibrate_confidence`（`backend.py:73-74`）；`temperature_scaling` 在策略目录（`postprocess_strategies.yaml:195-210`） |
| metric namespace | OK | 7 个 Literal（`agents/pareto.py:10-18`）；前缀规则 `removesuffix("_inference")`（`policy.py:143`）；Pareto 按 namespace 分区永不合并（`pareto.py:107-116`）；四重隔离强制点全部存在 |

**重点结论核实**：inference 候选不消耗 ASHA training budget——多级结构性强制成立（`NON_TRAINING_DOMAINS` → recommendation_only → executable-only 入 ASHA → `register_trial` 硬报错 → `inference_only_protocol`；`auto_optimization_loop.py:911,5043-5045,7891-7913,2253-2257`、`asha_scheduler.py:289,531`、`paper_protocol_catalog.py:105-129`）。

### 2.9 Custom Dataset（5/5 OK，3 处 FIXED）

| 子项 | 结论 | 证据 |
|---|---|---|
| data health | OK | 画像纯 label 文本统计不解码图像（`tools/dataset_stats.py:209`）；13 项 checklist 的每一项（越界/畸形/缺失/空标签/尺寸阈值 0.01-0.05/不平衡 0.2/弱指纹/train∩val）与代码逐项对应 |
| annotation | OK [FIXED×1] | 4 类 IssueType + 几何规则（`core/label_quality.py:18-39`）；`AnnotationAdvisor` 六类输出（`agents/annotation_advisor.py:73-95`）；全链路无写回 label；**FIXED**："without mutating data" docstring 引用归属从 `dataset_promotion.py` 修正为 `active_learning_stage_runner.py:89` |
| augmentation | OK [FIXED×1] | error→action 映射与 `configs/error_action_policies.yaml` 逐条一致；双轨执行（SAFE override 键集合 `auto_optimization_loop.py:876-909`）；两个反面例子属实；**FIXED**：9 契约 vs 12 具体类的口径澄清（`adapters.py:421-491` 实为 12 类，其中 9 个对应 `paper_data_adapters.yaml` 契约，另 3 个为非 paper-data 契约类） |
| leakage | OK [declared] | 泄漏防护止步于画像级弱指纹（`dataset_stats.py:692-699,727-728`）；preflight 不读 label 内容；`DatasetSplitPlanner` 零生产调用方（仅定义/re-export/测试）——文档如实声明，无夸大 |
| scene slices | OK [FIXED×1] | 固定 6 tag 不可扩展（`detection_error_profile_builder.py:51`）；`ap50` 恒为 None；缺失 tag 进 `unavailable_scene_slices`；GT 必须 COCO；**FIXED**：metadata JSON 示例的文件名键改为整数 image_id 键（代码 `int(key)` 解析，`detection_error_profile_builder.py:118-121`，文件名键会直接解析失败） |

**DatasetProfile vs DatasetReport 区分**：OK——`DatasetProfile` 为手填 metadata（`core/schemas.py:11-18`），`DatasetReport` 为自动统计（`tools/dataset_stats.py:119-135`），文档设专节区分。

### 2.10 Manifest 默认模式（附注核实，FIXED）

`custom-dataset.md` 原称 `metadata` 模式为默认——对 train/optimize preset 路径成立（`presets/coco_yolo26_auto.yaml`），但 `loop init`/`loop auto` 的 CLI 默认是 `sha256`（`cli.py:1348,1560`）。已改为按入口区分的准确表述。

---

## 3. 禁止冲突关键词扫描（11/11 通过）

逐词确认上下文仍然正确，非机械处理：

| 关键词 | 命中 | 上下文结论 |
|---|---|---|
| `scalar HPO disabled` | 4 文档处 + 2 历史审计 | 代码强制仍在：`auto_optimization_loop.py:3982-3999`（`scalar_hpo_disabled` 标记 + 拒绝）、`:3810`（`BudgetPolicy.allow_scalar_hpo`）；文档语义"HPO 仅存在于有界搜索"正确 |
| `83/85` | 5 处 | 全部为 frozen report metric 语境且标注 historical（`paper-83-campaign.md:19,76`、`paper-coverage-acceptance.md:13`、生成状态页、历史审计）；未冒充当前值 |
| `83/83` | 12 处 | 全部为 implementation ready / runtime preflight 语义（离线就绪审计）；`PRETRAINING_ACCEPTANCE.md` 明示 `REAL TRAINING EXECUTED: NO`；`training-readiness.md:51` 明示 "below 83/83 is a legitimate outcome" |
| `exact reproduction` | 9 处 | 全部为否定性声明（"not exact reproduction"）或 MethodProfile 边界说明；与 exact=0 的 artifact 状态一致 |
| `imgsz=640` | 20+ 处 | 全部为 invariant 语义；recipe 层硬校验仍在（`recipes/schemas.py:80,82,108` 三处抛错）；action space 声明 resolution action 的分层张力已有历史审计记录且文档表述为 recipe 层约束 |
| `datat`+`set`（历史 typo） | 交付文档 0 处 | 仅存于 memory 工作日志与 `DOCUMENTATION_AUDIT.md` 历史审计记录（验收规则豁免）；本报告自身曾因引用该词导致 typo lint 失败，已改为拆分写法——这也实证了 lint 在真实工作 |
| `incomplete` | 1 处 capability + 定义段 | `candidate_coco_error_facts` 为 `incomplete`（manifest 声明，生成器同源输出）；定义段解释准确 |
| `partial` | 1 处 capability + 定义段 | `error_delta_next_round` 为 `partial`；与代码现状相符 |
| `supported` | 无绝对化声明 | 未发现 "only supported/完整支持" 类无边界措辞；custom 场景矩阵均带 COCO 前置条件说明 |
| `executable` | 语境正确 | action execution class 语义（`executable` vs `recommendation_only`/`adapter_required`）；inference 文档中 3 处均为排除性叙述 |
| `optimize` | 1 处关键语境 | `docs/cli.md:200` 明示旧 optimize 子命令"供测试、迁移和维护使用，不是稳定的新手接口" |

---

## 4. 本轮修复（9 处）

全部为文档精度问题（数字/键名/引用/示例），不涉及行为声明真伪，未改动任何运行时代码：

1. `docs/architecture.md` — round cap 配置键名说明（YAML 键 `max_auto_rounds_safety`，Python 字段 `max_rounds_safety`）。
2. `docs/architecture.md` — 包行数 75k → 167k（实测 `yolo_agent` 包 167,106 行，排除 tests）。
3. `docs/coverage-baseline-summary.md` — 旧快照数字（80/80/47/2）整体过时，重写为当前生产 baseline（84/84/0、86、84）并以 report_hash 锚定。
4. `docs/cli.md` — `--run-id` 恢复语义区分"未完成工作自动恢复"与"已完成/停滞得到递增后缀"。
5. `docs/inference-policy-adapters.md` — action family 计数 26 → 27（`active_learning` 族）。
6. `docs/data-optimization.md` — scene-slice metadata JSON 示例键从文件名改为整数 image_id。
7. `docs/custom-dataset.md` — "without mutating data" docstring 引用归属修正。
8. `docs/custom-dataset.md` — manifest 默认模式按入口区分（train/optimize 为 `metadata`，loop init/auto 为 `sha256`）。
9. `docs/data-optimization.md` — data pipeline 组件口径澄清（9 契约 + 3 非 paper-data 契约类 = 12 个具体 adapter 类）。

---

## 5. 已声明限制（系统事实，非文档缺陷）

以下为**系统当前真实状态**，文档已如实声明且与 artifacts 一致，故不计为失败项：

- **无任何复现证据**：`pilot_reproduced_count = 0`、`exact_reproduction = 0`、`runtime_integrated` 及以上成熟度全 0、`real_training_executed: false`（全部 artifacts 一致）。
- **就绪 ≠ 结果**：83/83 implementation ready 与 83/83 runtime preflight 证明的是离线就绪，不是训练结果；`READY_FOR_FIRST_TRAINING` 未冒充已训练。
- **bounded HPO 尚未接入生产循环**：`plan_hpo_points` 无生产调用方，文档表述为 "bounded-by-construction, not yet an active loop behavior"。
- **active learning 为部分闭环**：mine → manifest → 外部 review → dataset-promote 决策成立；promote 不 mutate 数据、不触发重训。
- **data pipeline adapters 成熟度**：9 个 paper-data 契约全部 `adapter_implemented`（非 certified），启用需 paper route 证据。
- **DatasetSplitPlanner**：代码存在但零生产调用方，文档未声称具备自动 split 防护。
- **历史口径**：85 分母仅存在于冻结历史验收报告（manifest 标 `acceptance_lineage_status: historical`）；当前生产分母为 84。
- **85 分母 vs 83 manifest vs 728 catalog 三个语境**：文档已用专节区分（`paper-83-campaign.md` 第 1-3 节），互不混淆。

---

## 6. 门禁结果

| 门禁 | 命令 | 结果 |
|---|---|---|
| Docs consistency suite | 8 个专项测试 + encoding/beginner 回归 | **162 passed**（修复后重跑） |
| 全量快速 | `pytest -q` | **零失败**（2654 passed, 25 skipped, 1041 deselected；验收报告入库后复跑同样零失败） |
| 全量含慢速 | `pytest -q --run-slow` | **零失败**（3684 passed, 36 skipped；验收报告入库后复跑同样零失败） |
| 静态检查 | `ruff check .` | **All checks passed!**（修复后重跑） |

### 6.1 门禁过程记录（含一次自证）

- 验收中途曾出现 1 个真实失败：`test_no_known_path_typos` 抓到**本验收报告初稿在关键词表格中包含字面 typo**——lint 在真实工作，已把该词改为拆分写法并复跑确认（这也正是 Prompt 10 建立 CI 的意义）。
- `.pytest_cache` 的 `lastfailed` 中另有 48 条为**化石记录**（测试已被重命名/删除，pytest 对"本次未收集"的旧失败记录永久保留）；10 条现存条目复跑 78 passed，全部通过。已清理化石缓存。
- 最终复跑（验收报告入库状态）：`pytest -q` 与 `pytest -q --run-slow` 的 `.pytest_cache/v/cache/lastfailed` 均不存在 = 两次全量 run 零失败。

---

## 7. 最终 Verdict

## **PASS_WITH_DECLARED_LIMITATIONS**

**理由**：63 个文档经四路逐项代码对照与 11 个关键词上下文审查，未发现任何"描述不存在功能"或"声明与代码行为相悖"的问题；本轮发现的 9 处精度偏差（数字过时、键名/引用错位、示例不可用、计数口径混杂）已全部修复并通过文档一致性测试。双语文档结构、命令、状态块、能力矩阵、链接五项完全同步。生成的状态块与架构快照均处于 fresh 状态，由 CI 测试强制。

**声明限制**：系统级事实（零复现证据、真实训练未执行、bounded HPO 未接入生产循环、active learning 部分闭环、data adapters 非 certified）均已在文档中如实声明并被 artifacts 证实——这些是产品当前阶段的真实状态，不是文档错误，因此不构成 FAIL，但任何使用者在引用本文档时必须连同第 5 节的限制一起阅读。

**无未解决的 blocker。**
