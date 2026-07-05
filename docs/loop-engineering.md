# Loop Engineering

Loop orchestrator 是状态机，不是脚本拼接。

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
- `BatchTuner`: 探测 batch 32/48/64/96，选择不改变 imgsz 的最高吞吐
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

