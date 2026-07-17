# Paper Intelligence

Paper Intelligence 是训练前构建的离线研究输入，不是训练过程中临时搜索论文的功能。

## 生产链

```text
sync metadata
-> deduplicate
-> classify
-> LLM component extraction
-> contract draft
-> compatibility review
-> recipe generation
-> frozen ResearchSnapshot
```

先同步并构建快照：

```powershell
yolo-agent research build-snapshot --root research --sync --year-from 2020 --extract-components
```

已有本地 metadata 时，可以完全离线重建：

```powershell
yolo-agent research build-snapshot --root research --extract-components
```

训练期间不会调用 PaperScout，也不会回退读取实时 registry。每个基础 run 和 child round 都绑定同一个 `snapshot_hash`；更新论文后生成的新快照只对新 run 生效。

## 可用性

空 registry 也会生成可校验快照，但状态明确为：

```text
paper_intelligence=unavailable
reason=empty_registry
```

此时规则训练仍可继续，但 DecisionContext 不会声称使用了论文候选、paper prior 或论文经验。完全没有快照时同样标记 unavailable，并绑定稳定的 unavailable hash，避免训练期间读取 live registry。

## 成熟度统计

快照把以下计数写入 manifest 和 `snapshot_hash`：

- `metadata_only`
- `adapter_implemented`
- `smoke_passed`
- `pilot_reproduced`

成熟度变化会产生新的快照 hash。`metadata_only` 组件只能进入 implementation request，低于 `smoke_passed` 的组件不能进入训练队列。

论文指标始终是 `paper_claim` 或 `paper_prior`。只有协议匹配并导入本地 evidence 的结果，才可以提升 reproduction maturity。
