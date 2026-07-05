# Evidence 和报告

YOLO Agent 的推荐必须基于 evidence。没有证据时，报告会写：

```text
No evidence, do not trust this result.
```

## Evidence Contract

默认 loop evidence 包括：

- `dataset_report`
- `label_quality_report`
- `smoke_result`
- `latency_ms`
- `map50`
- `recall`

缺失项会写入：

```text
runs/{run_id}/artifacts/evidence_status.json
```

## Node-level Metrics

候选对比使用 node-level evidence：

```text
runs/{run_id}/metrics_by_node.jsonl
```

每条 metric 都绑定 candidate、node、dataset version、split、source、verified 状态和 validator。

## Artifact Manifest

Stage 输出会记录到：

```text
runs/{run_id}/artifacts/artifact_manifest.jsonl
```

每条 artifact 包含 name、type、path、sha256、producer_stage、created_at 和 schema_version。

## 报告

```powershell
yolo-agent report --run runs/coco-yolo26n --out report.md
```

报告包含任务画像、数据诊断、候选模型、消融变量、指标表、最佳模型推荐、下一轮建议、风险和未验证项。

