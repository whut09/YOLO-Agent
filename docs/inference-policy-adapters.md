# Isolated Inference Policy Adapters

Inference-only paper methods are evaluated outside the training recipe and outside
the standard 640 metric namespace. They reuse a frozen detector checkpoint; they do
not create training nodes, consume ASHA training budget, or receive model-component
attribution.

## Supported policies

| Policy | Component | Metric namespace |
| --- | --- | --- |
| SAHI slicing | `inference.sahi_slicing` | `sliced_*` |
| Tiled multi-scale | `inference.tiled_multi_scale` | `tiled_multi_scale_*` |
| Test-time augmentation | `inference.test_time_augmentation` | `tta_*` |
| Confidence calibration | `inference.confidence_calibration` | `calibrated_*` |
| Class-aware thresholding | `inference.class_aware_thresholding` | `class_threshold_*` |
| Cross-view merge | `inference.merge_policy` | `merged_*` |

Every policy records its protocol hash, predictions, COCO metrics, latency,
throughput, peak VRAM, and merge statistics. Standard `imgsz=640` results remain in
`standard_640` and are never overwritten.

## Certification command

Create a policy YAML such as:

```yaml
policy_id: tta-yolo26n
kind: test_time_augmentation
device: "0"
scales: [0.8, 1.0, 1.2]
horizontal_flip: true
merge_policy: weighted_box_fusion
allow_cross_view_merge: true
```

Run the isolated evaluation explicitly:

```powershell
yolo-agent advanced certify-inference-policy `
  --workdir runs/certification/tta-yolo26n `
  --model yolo26n.pt `
  --images E:\dataset\coco\images\val2017 `
  --annotations E:\dataset\coco\annotations\instances_val2017.json `
  --config configs\my-tta-policy.yaml `
  --standard-metrics runs\baseline\coco_eval.json `
  --execute
```

Without `--execute`, the command writes a skipped report and does not call the
model. SAHI remains available through `advanced certify-sahi` because its optional
dependency and native slicing backend have a separate certification contract.

## YOLO26 boundaries

- The standard comparison input remains `imgsz=640`; TTA and tiled views publish
  changed-protocol metrics only.
- A one-to-one YOLO26 head receives no extra NMS by default.
- Cross-view NMS requires `allow_cross_view_merge: true` and is recorded as an
  inference policy change.
- NMM and weighted box fusion merge only predictions from multiple views or tiles.
- Paper claims remain priors. Only local certification artifacts enter a Pareto
  front.
