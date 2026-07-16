# Loop Engineering

Loop orchestrator 是状态机，不是脚本拼接。
## Paper Recipe Round

自动训练入口 `yolo-agent train ...` 在每个自动轮次中会自动执行 Paper Intelligence：读取上一轮 evidence，生成 COCO error facts，查询本地论文 registry，并过滤 compatibility 和 component maturity。随后所有上下文只进入一次统一的 doctor-style LLM 决策；新人不需要额外记忆论文查询或 recipe 命令。

每轮唯一决策链是：`DecisionContext -> LLMDecisionBundle -> RecipeCritic -> Utility/Budget/Ablation -> RoundExecutionPlan`。LLM 负责诊断和提出候选；确定性 gate 决定候选是否能执行。LLM 成功时不会再混入另一套规则或论文 LLM 选择；LLM 跳过或失败时，系统才使用同一 `DecisionContext` 中记录的 deterministic fallback。

## Frozen Research Snapshot

论文智能层在训练前生产，不在训练轮次中临时联网：

```text
research sync
  -> deduplicate
  -> classify
  -> extract components
  -> contract draft
  -> YOLO26 compatibility review
  -> recipe generation
  -> reproduction queue
  -> frozen ResearchSnapshot
```

使用：

```powershell
yolo-agent research build-snapshot --root research
```

需要更新论文元数据时才显式同步：

```powershell
yolo-agent research build-snapshot --root research --sync --year-from 2020
```

需要在生产阶段调用本地配置的 LLM 提取论文组件时，显式增加 `--extract-components`。LLM 调用只发生在 snapshot build，不会发生在训练轮次：

```powershell
yolo-agent research build-snapshot --root research --extract-components
```

快照保存在 `research/snapshots/<snapshot_hash>/`，包含所有实际输入文件的副本和 SHA 校验。每个训练 run 会在 `run_context.yaml`、`paper_recipe_plan.yaml`、`DecisionContext` 和 `llm_decision_bundle.yaml` 中记录同一个 `research_snapshot_hash`。子轮次只继承这个 hash；如果快照内容被替换，Paper Intelligence 会停止并报告 hash 不一致，而不会悄悄使用新的论文或 recipe。

这一轮的详细文件包括：

- `paper_recipe_plan.yaml`：论文、组件和 recipe 候选上下文，不再单独调用 LLM。
- `llm_decision_bundle.yaml`：本轮唯一 LLM 诊断、proposal、critic 结果及最终执行映射。
- `round_execution_plan.yaml`：经过确定性 gate 后唯一有权生成执行队列的计划。
- component_compatibility.yaml：组件 maturity、adapter 和 imgsz=640 兼容快照。
- reproduction_state_*.yaml：每个论文组件的可恢复复现状态。
- decision_ledger.jsonl：输入摘要、prompt hash、模型输出、拒绝原因和最终决策。

终端仍只显示当前自动轮次、阶段、诊断、recipe、changed variable、训练进度和最终结论；机器细节留在这些 artifact 中。

## 执行边界

```text
ExperimentNode -> CommandSpec -> ExecutionResult -> EvidenceStore
```

`loop enqueue` 会在执行前，把计划好的 `ExperimentNode` 物化为可恢复的队列：

```text
ExperimentPlan -> ExecutionQueue -> Executor -> ExecutionResult -> EvidenceStore
```

## Executor

- `DryRunExecutor`: 只记录将要运行什么
- `ShellExecutor`: 对受控命令进行 subprocess 执行
- `UltralyticsExecutor`: 保守的 Ultralytics smoke/草案执行器
- `UltralyticsTrainExecutor`: typed `yolo detect train ...` 训练执行器
- `RuntimeProfiler`: 采集 GPU、it/s、epoch time、batch、cache 等运行证据
- `BatchTuner`: 从基础 batch 32/48/64/96 开始，并按可见 GPU 显存自动扩展候选；在不改变 `imgsz` 的前提下选择最高吞吐且不 OOM 的 batch
- `BenchmarkImporter`: 导入外部 benchmark 或 Ultralytics run 目录

## 持久化文件

```text
runs/{run_id}/run_context.yaml
runs/{run_id}/loop_state.yaml
runs/{run_id}/events.jsonl
runs/{run_id}/execution_queue.yaml
runs/{run_id}/artifacts/artifact_manifest.jsonl
runs/{run_id}/artifacts/decision_ledger.jsonl
runs/{run_id}/artifacts/execution_results/
runs/lineage.jsonl
```

## Stage Contract

Stage 顺序由 `configs/loop_policy.yaml` 定义。每个 stage 可以声明：

- `requires`
- `provides`
- `evidence_required`
- `block_on_missing`
- `retry_policy`
- `producer_artifacts`
- `artifact_contract`

缺少必要 evidence 的 stage 会进入 `blocked`，run 可以 resume，而不是继续产出不可信推荐。

## Next Round

`next_round.yaml` 基于 delta，而不是复制 checklist。它记录 parent run、当前最佳 evidence、未解决诊断、新补齐证据、推荐下一 stage 和停止原因。
