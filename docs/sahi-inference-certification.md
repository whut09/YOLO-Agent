# SAHI Independent Inference Certification

SAHI slicing is an inference policy, not a training recipe. It evaluates an existing
YOLO26 checkpoint with tiled images and publishes a separate metric namespace. It
does not modify the checkpoint, model graph, loss, dataloader, or training evidence.

## Install

```powershell
python -m pip install -e ".[sahi]"
```

The core package remains usable without SAHI. A missing optional dependency produces
a structured `skipped` certification report rather than silently running standard
inference.

## Run

The command is advanced and opt-in:

```powershell
yolo-agent advanced certify-sahi `
  --workdir runs/certification/sahi-yolo26n `
  --model yolo26n.pt `
  --images E:\dataset\coco\images\val2017 `
  --annotations E:\dataset\coco\annotations\instances_val2017.json `
  --device 0 `
  --slice-height 640 `
  --slice-width 640 `
  --overlap-height 0.2 `
  --overlap-width 0.2 `
  --merge-policy none `
  --standard-metrics runs\baseline\standard_metrics.json `
  --execute
```

Without `--execute`, the command only writes a safe `skipped` report. The default
`merge-policy=none` adds no NMS to the YOLO26 one-to-one path. `nms` or `nmm` may be
selected explicitly for cross-slice merging; that choice is recorded in the protocol
hash and does not alter standard YOLO26 inference.

## Evidence Boundaries

Standard 640 metrics keep their names under `standard_640_metrics`. Sliced inference
uses only:

- `sliced_map50_95`
- `sliced_ap_small`
- `sliced_latency_ms`
- `sliced_throughput`

The certification report sets `inference_policy_changed=true` and
`training_attribution_allowed=false`. Standard and sliced observations enter separate
Pareto fronts. A sliced gain cannot be attributed to a training component, cannot
promote an ASHA training candidate, and cannot overwrite the standard 640 result.

Artifacts are written under the selected work directory:

```text
artifacts/slicing_inference_protocol.json
artifacts/sliced_predictions.json
artifacts/sliced_metrics.json
sahi_certification_report.yaml
```

The report records the optional dependency version, protocol hash, isolation checks,
metric namespaces, and artifact paths. A failed check produces no reproduction claim.
