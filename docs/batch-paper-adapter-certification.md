# Batch Paper Adapter Certification

`PaperAdapterCertificationFactory` 批量发现并认证所有真实、可复用的 `ComponentAdapter`。发现过程检查实际 Python 类型，不使用 component ID 前缀判断 adapter 是否存在，并排除 `DummyAdapter` 测试夹具。

## CPU Batch

默认命令只执行 CPU 认证，不运行真实 GPU 训练：

```powershell
yolo-agent advanced certify-paper-adapters --cpu
```

每个 adapter 独立执行：

```text
adapter import
-> typed runtime payload
-> hook signature validation
-> unit validation
-> isolated local CPU smoke
```

一个 adapter 失败不会阻止后续 adapter。失败 report、worker log 和 maturity artifact 会保留，但失败 artifact 不会提升 maturity。

## Resume And Changed-only

中断后继续未完成或失败的 adapter：

```powershell
yolo-agent advanced certify-paper-adapters --cpu --resume
```

只认证新增或 runtime identity 已变化的 adapter：

```powershell
yolo-agent advanced certify-paper-adapters --cpu --changed-only
```

resume 只有在 batch report hash、单 adapter report、adapter hash、Ultralytics version 和 protocol hash 全部有效时才复用通过结果。代码 commit 作为 provenance 保留，但无关文档或其他 adapter 的 commit 不会单独触发重跑；adapter source、Ultralytics 或 certification protocol 变化都会自动重新认证。

可重复使用 `--component` 限制本次范围：

```powershell
yolo-agent advanced certify-paper-adapters --cpu `
  --component sampling.small_object `
  --component loss.quality.correlation
```

## GPU Batch

GPU 必须同时指定模式和执行许可：

```powershell
yolo-agent advanced certify-paper-adapters --gpu --execute-real-gpu `
  --model yolo26n.pt --data coco.yaml --device 0
```

每个 adapter 先复验 CPU smoke，再执行真实 GPU single-batch/train hook、backward、AMP、checkpoint 和 resume 验证。通过后生成 matched pilot fixture，冻结 candidate/control 的 model、dataset fixture、seed、epochs、batch、`imgsz=640`、Ultralytics version 和 eval protocol。

Distillation 可以显式提供本地 teacher：

```powershell
yolo-agent advanced certify-paper-adapters --gpu --execute-real-gpu `
  --teacher yolo26s.pt --ensemble-teacher yolo26m.pt
```

Matched pilot fixture 只是下一阶段的公平比较协议，不包含 paired delta，也不是本地收益证据。batch GPU 认证的成熟度上限是 `gpu_certified`；`pilot_reproduced` 仍要求真实 matched pilot、COCO post-eval 和 paired evidence。

## Artifacts

默认输出：

```text
runs/certification/paper-adapters/
  paper_adapter_certification.yaml
  paper_adapter_coverage.yaml
  <component_id>/
    component_certification.cpu.yaml
    component_certification.gpu.yaml
    batch_result.yaml
    matched_pilot_fixture.yaml
```

认证成功后 factory 原子更新 `runs/component_maturity_registry.yaml`，再根据有效 overlay 生成 machine-local coverage report。coverage 刷新失败会在 batch report 中单独记录，不会删除已经生成的 adapter evidence。
