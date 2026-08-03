# YOLO26 Graph Components

YOLO Agent implements reusable graph mechanisms as isolated YOLO26 adapters. It
does not copy complete detector architectures, and component adaptation is not
claimed as exact paper reproduction.

## Runtime Boundaries

`head.p2_small_object` adds a stride-4 feature path and feeds four scales into
the native YOLO26 Detect head and loss. The remaining graph adapters transform
the runtime-detected P3/P4/P5 tensors immediately before native Detect while
preserving strides 8/16/32 and their channel contract.

| Component | Runtime mechanism | Insertion point |
| --- | --- | --- |
| `neck.weighted_feature_pyramid` | Learnable cross-scale weights | Before Detect P3/P4/P5 |
| `neck.bidirectional_feature_fusion` | Top-down and bottom-up fusion | Before Detect P3/P4/P5 |
| `neck.gold_gather_distribute` | Gather-distribute fusion | Before Detect P3/P4/P5 |
| `neck.rtmdet_large_kernel` | Isolated large-kernel depthwise block | Before Detect P3/P4/P5 |
| `neck.lightweight` | Lightweight depthwise neck | Before Detect P3/P4/P5 |
| `block.reparameterized_convolution` | Training branches with audited deploy fusion | Before Detect P3/P4/P5 |
| `attention.channel` | Channel-wise feature weighting | Before Detect P3/P4/P5 |
| `attention.spatial` | Spatial feature weighting | Before Detect P3/P4/P5 |
| `neck.deformable_feature_aggregation` | Explicit local deformable operator | Before Detect P3/P4/P5 |

`neck.multi_scale_fusion` remains as the backward-compatible generic fusion
identity. Specific paper evidence should select a specific mechanism instead.
Generic terms such as `attention` do not authorize an adapter.

## Evidence And Guards

Every runtime writes a hash-bound manifest containing the component and adapter
identity, mechanism configuration, input/output strides and channels, plugin
call evidence, partial checkpoint matched/missing/new keys, and resource deltas.
The runtime verifies real forward, native loss, backward, AMP, and export dry-run
behavior with fixed `imgsz=640`.

YOLO26 end-to-end one-to-one/one-to-many behavior remains native. These adapters
do not add external NMS and do not restore DFL. Latency, VRAM, parameter count,
and model size are hard gates for certification and promotion.

Deformable aggregation never falls back to an ordinary convolution. If its
configured local operator cannot be imported, the result is an
`implementation_request`, not an executable recipe.

## Certification

Source contracts remain conservatively marked `adapter_implemented`. Validate a
component locally before it can be considered for a paper recipe:

```powershell
yolo-agent advanced certify-component --component attention.channel --cpu
yolo-agent advanced certify-component --component attention.channel --gpu
```

GPU certification is explicit and requires the CPU artifact chain first. CPU or
GPU certification does not claim `pilot_reproduced`; that maturity requires a
matched candidate/control pilot, post-evaluation, paired delta, and ASHA result.

