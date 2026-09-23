# Codex Prompts: Paper Candidate Routing

这组 prompt 用于修复“论文方法已经存在，但没有进入真实训练队列”的架构问题。请按顺序执行，每段完成并提交后再执行下一段。每段都应在 `E:\\codex\\YOLO-Agent` 工作区内操作。

## 先统一目标

不要把“所有论文都会训练”作为不诚实的承诺。系统必须保证：

1. 每个与 YOLO26、当前数据集、当前目标和固定 `imgsz=640` 兼容的论文提案，都会进入候选台账。
2. 已经 runtime-ready、证据完整且未测试的候选，必须进入 ASHA trial；不能在 planner、critic、materialization 或 ASHA 注册之间静默消失。
3. 尚不能训练的提案必须保留一个明确 disposition：`queued`、`already_tested`、`evidence_recovery`、`implementation_request`、`incompatible`、`blocked_runtime` 或 `deferred_budget`。
4. `deferred_budget` 只能表示预算顺延，不能表示候选被丢弃；下一轮必须可以恢复。
5. 论文年份、LLM 建议或 paper claim 不能覆盖 YOLO26 不兼容性、缺失证据、已有负结果或 matched-control 约束。
6. 训练结果只有在 candidate 和 matched baseline 都完成、协议和 split 一致时才允许产生 mAP delta。

当前整体 mAP 诊断下，首批必须可追踪以下组件：

`loss.hard_negative_classification`、`sampling.hard_negative_replay`、`loss.quality.correlation`、`loss.quality.pseudo_iou`、`assigner.task_aligned`、`assigner.optimal_transport`、`distillation.yolo26_teacher_student`、`neck.rtmdet_large_kernel`。

小目标专用方法不应因为“整体 mAP”目标而占用首批队列，但必须在台账中记为 `incompatible` 或 `deferred_budget`，不能消失。

---

## Prompt 1：先做架构审计和 canonical ID 收敛

```text
在 E:\\codex\\YOLO-Agent 内工作。先不要大规模重构，完成一次可审计的候选路由审计。

阅读并画出实际调用链：
PaperMethodProfile/diagnosis -> PaperRecipePlanner -> LLM/recipe critic ->
PaperRecipeMaterializationGate -> RoundExecutionPlan -> _register_guarded_pilot_trials -> ASHA。

重点检查：
- yolo_agent/agents/paper_recipe_planner.py
- yolo_agent/agents/paper_recipe_materialization/gate.py
- yolo_agent/agents/paper_recipe_materialization/schemas.py
- yolo_agent/agents/auto_optimization_loop.py
- yolo_agent/research/component_aliases.py
- configs/component_aliases.yaml
- configs/paper_diagnosis_rules.yaml
- configs/recipes/*.yaml
- configs/components/**/*.yaml

建立一个统一的 canonical component resolver。至少验证并覆盖：
- sampling.hard_negative -> sampling.hard_negative_replay；不要把两者当成两个不同机制
- loss.quality.correlation 和 loss.quality.pseudo_iou 是两个可执行候选，不要被抽象的
  loss.quality.iou_aware_classification 取代
- assigner.task_aligned、assigner.optimal_transport 必须保留为独立 canonical ID
- distillation.yolo26_teacher_student 和 neck.rtmdet_large_kernel 必须能从 paper profile
  映射到同名 runtime contract

不要用字符串 contains 代替 canonical resolver，也不要把未知 ID 自动映射到相似组件。
未知 ID 必须返回明确的 unresolved 结果，交给后续 implementation_request。

补充单元测试，覆盖 alias、未知 ID、抽象机制到多个具体候选的 one-to-many 映射，
并确保 recipe、component contract、method profile 三者的 canonical ID 不一致时测试失败。
只运行相关 CPU 测试和 Ruff。完成后提交：
git add <本段实际修改的文件>
git commit -m "Unify canonical paper component routing"
```

## Prompt 2：建立“不丢候选”的 Proposal Disposition Ledger

```text
在 E:\\codex\\YOLO-Agent 内继续工作。实现一个持久化的 paper candidate disposition ledger，
不要只依赖现有 decision_ledger 的零散事件。

建议新增：
- yolo_agent/agents/paper_proposal_ledger.py
- yolo_agent/agents/paper_proposal_schemas.py

每条记录至少包含：run_id、round_index、paper_ids、method_profile_ids、recipe_id、recipe_version、
canonical_component_ids、combination_id、execution_fingerprint、source_stage、disposition、reason_codes、
required_evidence、required_adapters、matched_error_fact_ids、budget_rank、created_at。

disposition 只能是：
queued、already_tested、evidence_recovery、implementation_request、incompatible、blocked_runtime、deferred_budget。

把 ledger 写入 runs/<run>/artifacts/paper_candidate_coverage.yaml，并在每个边界写入：
1. planner 看见的全部 recipe/profile proposal；
2. critic 的 accepted/rejected 结果；
3. materialization 的每个 candidate input；
4. RoundExecutionPlan 的每个 node；
5. ASHA 注册成功或失败的每个候选。

实现 invariant：同一 execution_fingerprint 在同一协议内只能有一个训练候选，但一个候选可以覆盖多个 paper_id；
任何 stage 丢弃候选都会使测试失败。禁止用“未 selected”作为无记录的终态。

增加 fixture：构造 8 个 smoke/runtime-ready 组件和 2 个有效 coupled recipe，验证 10 个候选都出现于 ledger，
即使其中一部分因为预算被标为 deferred_budget。

只运行 CPU 测试。提交：
git add <本段实际修改的文件>
git commit -m "Add auditable paper candidate disposition ledger"
```

## Prompt 3：补齐 diagnosis -> recipe 的五条路由

```text
在 E:\\codex\\YOLO-Agent 内继续工作。修复诊断事实到 recipe 的绑定遗漏，目标是整体 mAP，不能把所有问题归因到 small_object。

修改并测试：
- configs/paper_diagnosis_rules.yaml
- configs/recipes/yolo26_quality_alignment.yaml
- configs/recipes/yolo26_data_pipeline.yaml
- configs/recipes/yolo26_assignment_shadow.yaml
- configs/recipes/yolo26n_distillation.yaml 或实际加载的 distillation recipe
- configs/recipes/yolo26_multi_scale_necks.yaml
- yolo_agent/agents/paper_recipe_planner.py
- yolo_agent/agents/recipe_critic.py

必须新增或修复以下绑定：
- background_false_positive_class、high_confidence_false_positive、class_confusion_pair
  -> loss.hard_negative_classification 和 sampling.hard_negative_replay
- confidence_localization_mismatch、localization_error、localization_heavy_class
  -> loss.quality.correlation 和 loss.quality.pseudo_iou
- assignment_conflict、duplicate_prediction、confidence_localization_mismatch
  -> assigner.task_aligned 和 assigner.optimal_transport
- class_low_ap、representation_gap、capacity_gap
  -> distillation.yolo26_teacher_student
- localization_error、scale_variation、feature_relation_gap
  -> neck.rtmdet_large_kernel

不要把 loss.quality.iou_aware_classification 自动当成 correlation 或 pseudo_iou；它只能作为抽象 proposal，
如果没有明确实现绑定就进入 implementation_request。

matching_error_facts 必须支持一个 recipe 绑定多个 fact pattern，并在 ledger 中记录具体命中的 fact ID。
对整体 mAP 目标，small-object-only recipe 不得成为唯一候选来源。

增加一个 improve-map-11 diagnosis fixture，断言上述 8 个组件至少都被 planner 看到并得到 disposition。
提交：
git add <本段实际修改的文件>
git commit -m "Bind diagnosis facts to overall mAP paper recipes"
```

## Prompt 4：实现 split-safe hard-negative evidence

```text
在 E:\\codex\\YOLO-Agent 内继续工作。实现 hard-negative replay 的证据生产链，不允许只改 recipe YAML。

阅读：
- yolo_agent/tools/coco_error_mining.py
- yolo_agent/tools/coco_error_importer.py
- yolo_agent/core/error_facts.py
- yolo_agent/components/adapters/data_pipeline/*
- configs/recipes/yolo26_data_pipeline.yaml

新增一个稳定的 hard-negative manifest schema，至少保存：dataset_manifest_hash、source_split、image_id、
sample_index、predicted_class、score、bbox、error_type、source_run_id、baseline_protocol_hash、manifest_hash。

要求：
- 训练 replay manifest 只能来自 train split 或经过明确的 split-safe 映射；不能把 validation annotation 直接泄漏到训练。
- 当前 baseline 没有足够的 train-side hard-negative evidence 时，候选 disposition 必须是 evidence_recovery，
  并自动生成 evidence recovery action，而不是静默跳过。
- manifest 必须被 data adapter 的 runtime payload 引用，并在 sampler 中验证 hash、split、sample index 合法。
- sampling.hard_negative_replay 与 loss.hard_negative_classification 可以分别做 atomic candidate，
  也要为后续 coupled candidate提供稳定的 evidence ID。

增加 CPU fixture 测试：空 manifest、跨 split manifest、重复样本、协议不一致、有效 replay manifest。
验证有效 manifest 能生成 candidate policy 和 dataloader plugin payload。
提交：
git add <本段实际修改的文件>
git commit -m "Add split-safe hard-negative evidence pipeline"
```

## Prompt 5：让 quality loss 成为真实候选

```text
在 E:\\codex\\YOLO-Agent 内继续工作。把 loss.quality.correlation 和 loss.quality.pseudo_iou 从
“component 已 smoke_passed”打通到真实 candidate。

阅读：
- yolo_agent/components/adapters/losses/quality_alignment.py
- yolo_agent/certification/quality_loss.py
- yolo_agent/components/adapters/audit_contract.py
- configs/components/loss/quality_alignment.yaml
- configs/recipes/yolo26_quality_alignment.yaml
- yolo_agent/agents/auto_optimization_loop.py

要求：
- 两个 component 必须各有独立 recipe、changed_variable、runtime plugin、payload 和 evidence artifact。
- planner 命中 localization_error 或 confidence_localization_mismatch 时，两个 recipe 都要进入 ledger；
  由 budget/ASHA 决定顺序，不能在 planner 阶段二选一后丢失另一个。
- candidate policy 必须保留 primary metric map50_95、localization error metric、latency 和 model size guards。
- 不允许把质量损失改成 bbox regression replacement，也不允许恢复 YOLO26 DFL。
- 训练 hook 异常必须成为 candidate failed，并写 traceback artifact；不能让整个 CLI 打出未处理 traceback。

用 mock backend 测试两个候选均能注册 ASHA、生成 matched baseline、写 paired result。
增加 negative test：缺 runtime payload、缺 changed variable、fixed imgsz 非 640 时不得入队。
提交：
git add <本段实际修改的文件>
git commit -m "Route quality alignment losses into ASHA candidates"
```

## Prompt 6：完成 assignment shadow -> active 的状态机

```text
在 E:\\codex\\YOLO-Agent 内继续工作。修复 assigner.task_aligned 和 assigner.optimal_transport 只有 shadow、没有真实 active pilot 的问题。

阅读：
- configs/recipes/yolo26_assignment_shadow.yaml
- yolo_agent/certification/assignment_shadow.py
- yolo_agent/components/adapters/assigners/yolo26_assignment.py
- yolo_agent/agents/auto_optimization_loop.py
- yolo_agent/agents/asha_scheduler.py
- yolo_agent/core/round_execution_plan.py

实现明确状态：shadow_planned -> shadow_evidence_complete -> active_candidate_eligible -> active_pilot -> promoted/rejected。

要求：
- shadow 只产生 assignment metrics，不冒充 mAP improvement。
- shadow 满足 minimum batches、positive assignment valid、native loss equivalence、latency/memory guards 后，
  自动生成同一 canonical component 的 active recipe，mode=active，保留 matched control 和 protocol hash。
- task_aligned 与 optimal_transport 必须是两个独立 ASHA trial；一个 shadow 结果不能吞掉另一个。
- shadow 未通过时 disposition=blocked_runtime 或 evidence_recovery，reason 必须具体。
- active candidate 必须进入 RoundExecutionPlan 和 ASHA，不能要求用户额外运行 certification 命令。

增加 CPU mock tests 覆盖两条状态机、active candidate 注册、shadow 未通过、重复 assignment recovery。
提交：
git add <本段实际修改的文件>
git commit -m "Promote assignment shadow evidence to active pilots"
```

## Prompt 7：打通 teacher-student distillation

```text
在 E:\\codex\\YOLO-Agent 内继续工作。让 distillation.yolo26_teacher_student 能作为整体 mAP candidate 真实训练，
并保持最终 student 仍为 yolo26n。

阅读：
- yolo_agent/components/adapters/distillation/yolo26_distillation.py
- yolo_agent/certification/distillation.py
- configs/components/distillation/yolo26_teacher_student.yaml
- configs/recipes/yolo26n_distillation.yaml
- tests/test_yolo26_distillation.py

要求：
- teacher 只允许 frozen checkpoint；student、dataset split、imgsz、protocol 必须显式绑定。
- 自动解析 teacher checkpoint；不存在、hash 不匹配、student/teacher 数据 split 不一致时，
  disposition=blocked_runtime 或 evidence_recovery，禁止静默跳过。
- runtime payload 和 evidence 必须写 teacher/student path、sha256、dataset hash、split、loss mode。
- teacher 只参与训练，导出和最终 model_size/latency 只测 student。
- planner 命中 class_low_ap、representation_gap 或 capacity_gap 时，候选必须进入 ledger；
  缺 teacher 时也要保留 implementation/evidence disposition。

增加 CPU tests：teacher resolution、hash mismatch、same split、student-only export、ASHA registration。
用 mock backend 验证 candidate 和 matched baseline 都完成后才产生 paired mAP delta。
提交：
git add <本段实际修改的文件>
git commit -m "Make YOLO26 teacher student distillation trainable"
```

## Prompt 8：打通 RTMDet large-kernel neck

```text
在 E:\\codex\\YOLO-Agent 内继续工作。让 neck.rtmdet_large_kernel 成为真实 model-graph candidate。

阅读：
- yolo_agent/components/adapters/neck/rtmdet_adapter.py
- yolo_agent/components/adapters/neck/rtmdet_large_kernel.py
- yolo_agent/components/adapters/neck/runtime.py
- yolo_agent/certification/neck_graph.py
- configs/components/neck/yolo26_multi_scale.yaml
- configs/recipes/yolo26_multi_scale_necks.yaml

要求：
- plugin 只替换允许的 neck 节点，保持 YOLO26 one-to-one head、native DFL-free regression 和 imgsz=640。
- candidate payload 必须包含 graph identity、input/output shape contract、adapter version/hash、rollback plan。
- planner 命中 localization_error、scale_variation 或 feature_relation_gap 时，recipe 必须入 ledger。
- 自动做 CPU shape/forward/build smoke；不要要求用户手工运行 certification 命令。
- candidate 记录 latency_ms、peak_vram_mb、model_size_mb；模型图变化导致 guard 失败时是 blocked_runtime 或 rejected，
  不能被标记为“没有候选”。

增加 mock graph tests 和一个 mock ASHA candidate registration test。
提交：
git add <本段实际修改的文件>
git commit -m "Route RTMDet large kernel neck into paper trials"
```

## Prompt 9：只生成有证据的 coupled paper combinations

```text
在 E:\\codex\\YOLO-Agent 内继续工作。修复 coupled recipe 被 critic 或 auto loop 丢弃的问题，并禁止无约束笛卡尔积。

阅读：
- yolo_agent/recipes/coupled_library.py
- yolo_agent/recipes/schemas.py
- yolo_agent/agents/recipe_ablation_planner.py
- yolo_agent/agents/paper_recipe_materialization/gate.py
- configs/coupled_recipe_templates.yaml

实现 explicit combination generator，只从 template、MethodProfile coupling_reason 或 verified local diagnosis 生成组合。
至少支持：
- loss.hard_negative_classification + sampling.hard_negative_replay
- neck.rtmdet_large_kernel + loss.quality.correlation
- neck.rtmdet_large_kernel + loss.quality.pseudo_iou
- assigner.task_aligned + loss.quality.correlation/pseudo_iou（仅在 assignment shadow passed 后）
- assigner.optimal_transport + loss.quality.correlation/pseudo_iou（仅在 assignment shadow passed 后）
- distillation.yolo26_teacher_student + sampling.class_balanced（仅在 class imbalance/capacity gap 证据存在时）

每个 coupled candidate 必须自动生成 baseline、arm A、arm B、A+B；每个 arm 都有 matched control，
并在 ledger 中记录 component IDs、paper IDs、coupling reason、internal ablation plan、combination fingerprint。

不要把 CoupledRecipe 排除在 executable_pilot_policies 之外；不要把 coupled candidate 拆成一个不带组合语义的 atomic candidate。
没有 coupling evidence 时 disposition=implementation_request 或 incompatible，并写明原因。

增加组合生成、四臂 ablation、重复组合和缺失 shadow evidence 的 CPU tests。
提交：
git add <本段实际修改的文件>
git commit -m "Materialize evidence-bound paper combinations"
```

## Prompt 10：按 execution fingerprint 去重，不按 paper 数量去重

```text
在 E:\\codex\\YOLO-Agent 内继续工作。审计并统一所有 candidate fingerprint，避免“同一实现被多个论文覆盖时重复训练”，
也避免“一个论文重复出现导致其他独立实现被吞掉”。

阅读：
- yolo_agent/agents/asha_scheduler.py
- yolo_agent/agents/paper_candidate_orchestrator.py
- yolo_agent/agents/paper_recipe_materialization/candidate_priority.py
- yolo_agent/agents/exploration_diversity.py
- yolo_agent/core/policy_memory.py

execution_fingerprint 必须包含：model checkpoint identity、canonical component IDs、recipe/version、effective overrides、
dataset manifest hash、baseline protocol hash、imgsz、fidelity、seed、teacher/graph/runtime payload hashes、组合 IDs。

规则：
- 相同 fingerprint：保留一个 trial，合并 paper_ids 和 method_profile_ids，ledger=already_tested 或 queued。
- 不同 component、不同 override、不同 teacher、不同 graph 或不同 coupled combination：不得去重。
- paper year、paper_id 不得单独决定是否重复训练。
- 已完成且 paired evidence valid 的 fingerprint=already_tested；只有旧 debug 或 unmatched evidence 不能算 already_tested。

增加回归测试覆盖 alias、同实现多论文、不同实现同论文、旧 artifact protocol mismatch。
提交：
git add <本段实际修改的文件>
git commit -m "Deduplicate candidates by execution fingerprint"
```

## Prompt 11：让 ASHA 注册整个 eligible cohort

```text
在 E:\\codex\\YOLO-Agent 内继续工作。修复“planner 有候选但 ASHA registered=0”以及“默认最多 6 个候选导致其他 paper candidate 消失”。

阅读：
- yolo_agent/agents/paper_recipe_planner.py
- yolo_agent/agents/auto_optimization_loop.py，重点是 _ensure_paper_intelligence、_register_guarded_pilot_trials
- yolo_agent/agents/asha_scheduler.py
- yolo_agent/core/round_execution_plan.py
- yolo_agent/agents/budget_optimizer.py

要求：
- PaperRecipePlanner 先完整产生 candidate ledger，不得用 BudgetOptimizer(max_candidates=6) 删除未选候选。
- budget optimizer 只能决定当前 round 的 assignment 顺序；其余 eligible candidate 必须进入 deferred_budget，
  并保留可恢复的 ASHA trial。
- atomic 和 coupled 都必须进入 deferred_nodes；不能只收 AtomicRecipe。
- 每个 eligible candidate 都必须有 matched baseline control；缺 control 是明确 blocked_runtime，不能导致整个注册函数返回 0。
- target_error_facts 为空必须是 evidence_recovery，并记录需要的 fact，不得静默 continue。
- overall mAP 目标下，先覆盖 hard-negative、quality、assignment、distillation、neck 五类，再允许 patience stop；
  不要让 4 个 native augmentation 负结果耗尽 paper cohort。
- 如果 executable candidate 数量大于单轮预算，注册全部但按 ASHA allocation 分批训练；CLI 必须显示 queued/deferred 数。

增加 improve-map-11 fixture：断言上述 8 个 atomic candidates 和有效 coupled candidates 的 ASHA trial 数量 > 0，
且 candidates_planned > 0 时绝不会出现 ASHA_trials_registered=0，除非所有候选都有逐条 terminal disposition。
提交：
git add <本段实际修改的文件>
git commit -m "Register the full eligible paper cohort with ASHA"
```

## Prompt 12：把 runtime readiness 变成一键自动流程

```text
在 E:\\codex\\YOLO-Agent 内继续工作。保留运行时安全检查，但取消用户必须手工执行 certification 命令的流程。

阅读：
- yolo_agent/certification/component_queue_gate.py
- yolo_agent/certification/paper_adapter_factory.py
- yolo_agent/agents/auto_optimization_loop.py
- yolo_agent/agents/optimize_runner.py
- yolo_agent/cli.py

实现 AutomaticRuntimeReadinessGate：
- CPU contract/shape/forward smoke 在 candidate materialization 时自动执行并缓存；
- adapter/runtime payload/protocol hash 不变时复用缓存；
- 缓存缺失时由同一个 train 命令自动补做，不要求用户输入第二条命令；
- external GPU busy 与 candidate OOM 必须区分：前者 resource blocked，可恢复；后者 candidate failed，写失败 artifact；
- readiness failure 只阻塞该 candidate，不能使其他独立 paper candidates 消失；
- 不得把 readiness 检查伪装成 mAP 训练结果，也不得把 GPU certification 失败说成搜索完成。

增加 CPU/mock tests，验证一条命令能从 planner 到 readiness 到 ASHA registration；并验证一个候选失败时其他候选仍可排队。
提交：
git add <本段实际修改的文件>
git commit -m "Automate runtime readiness without manual certification"
```

## Prompt 13：收敛用户可读的终端摘要和异常处理

```text
在 E:\\codex\\YOLO-Agent 内继续工作。终端只展示用户决策所需信息，详细诊断留在 artifacts。

修改 yolo_agent/cli.py、yolo_agent/core/loop_status.py 及相关 summary builder，使 train/optimize 结束时固定输出：

状态：正常完成 / 正在训练 / 已阻塞 / 训练失败 / 搜索完成未达标
训练：baseline、candidate、coupled arm 是否真的训练；候选数量和已完成数量
结果：baseline mAP50-95、best candidate mAP50-95、verified delta；没有 paired result 就明确“未测得提升”
问题：唯一主 blocker，例如“候选需要 teacher checkpoint”或“GPU 被 PID 占用”
下一步：一条可执行命令或明确“代码需要先修复”；不要输出旧的 misleading Next command

异常处理要求：
- 用户终端不能出现未捕获 Python traceback；保存到 artifacts/logs/cli_exception.log，并显示路径；
- `Reason: complete` 不能用于 blocked、training_failed、no_registered_trials；
- 显示 paper candidate coverage：queued/already_tested/evidence_recovery/implementation_request/
  incompatible/blocked_runtime/deferred_budget；
- 详细 paper IDs、adapter hashes、ASHA 状态仍写 YAML/JSON artifact。

增加 CLI snapshot tests，覆盖：正常 paired result、GPU busy、candidate failed、ASHA zero registration、搜索完成未达标。
提交：
git add <本段实际修改的文件>
git commit -m "Make paper optimization results user-readable"
```

## Prompt 14：建立整体 mAP 的端到端 acceptance suite

```text
在 E:\\codex\\YOLO-Agent 内继续工作。新增一个不依赖真实 GPU 的端到端 acceptance suite，证明候选没有在中间层消失。

新增或扩展 tests/test_paper_candidate_routing_acceptance.py，使用 improve-map-11 diagnosis fixture 和 mock backend。

必须断言：
1. planner 看到并记录：
   loss.hard_negative_classification、sampling.hard_negative_replay、loss.quality.correlation、
   loss.quality.pseudo_iou、assigner.task_aligned、assigner.optimal_transport、
   distillation.yolo26_teacher_student、neck.rtmdet_large_kernel；
2. 每个候选在 ledger 中恰好有一个当前 disposition；
3. runtime-ready 且 evidence-complete 的候选进入 RoundExecutionPlan 和 ASHA trial；
4. coupled candidates 具有 baseline/A/B/A+B 内部消融，不会被 AtomicRecipe 过滤；
5. overall mAP 目标不会把 small-object-only 方法当成首批唯一候选；
6. candidate 与 matched baseline protocol/split 不一致时没有 paired delta；
7. 一个候选失败不会删除其他候选的 ledger 或 ASHA trial；
8. old run protocol mismatch 会进入 recovery/isolated-run disposition，而不是复用错误 evidence；
9. `candidates_planned > 0` 且存在可执行候选时，ASHA_trials_registered=0 必须使测试失败，除非每个候选都有逐条明确 blocker。

运行相关 CPU 测试、完整 pytest、Ruff、compileall、git diff --check。提交：
git add <本段实际修改的文件>
git commit -m "Add end-to-end paper candidate routing acceptance"
```

## Prompt 15：最终离线审计和交付

```text
在 E:\\codex\\YOLO-Agent 内做最终审计，不再新增功能。

运行：
- pytest -q
- ruff check .
- python -m compileall yolo_agent tests
- git diff --check

生成 docs/paper-candidate-routing-audit.md，内容必须包括：
- 当前调用链和每个候选 disposition 的定义；
- improve-map-11 fixture 的 candidate coverage 表；
- 8 个重点组件的 contract、recipe、runtime entrypoint、required evidence；
- 每个有效 coupled template 和 A/B/A+B 规则；
- 哪些候选可以自动训练，哪些仍会因为真实外部状态或缺 checkpoint 被阻塞；
- 明确不能承诺 +2 mAP，只能承诺候选不再静默丢失、有效 paired comparison 才能宣称提升。

检查工作树中没有生成的运行目录、临时日志或模型权重被误提交。最后创建一个命名提交：
git add <本段实际修改的文件>
git commit -m "Complete auditable paper candidate routing"
不要执行真实 GPU 训练；交付测试结果、最终 commit hash 和下一条用户训练命令格式。
```
