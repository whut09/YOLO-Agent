# Guarded Multi-scale Neck Plugins

YOLO-Agent provides three isolated `ModelGraphPlugin` implementations for the
native YOLO26 P3/P4/P5 feature boundary:

| Component | Source | Isolated change | Local status |
| --- | --- | --- | --- |
| `neck.multi_scale_fusion` | Generic design | Bidirectional residual fusion | `smoke_passed` |
| `neck.gold_gather_distribute` | Gold-YOLO, arXiv:2309.11331 | Gather, context mix, gated distribution | `smoke_passed` |
| `neck.rtmdet_large_kernel` | RTMDet, arXiv:2212.07784 | 5x5 depthwise residual blocks | `smoke_passed` |

`smoke_passed` means the local adapter passed shape, backward, AMP, export,
checkpoint-audit, and runtime-hook tests. It does not mean the component has
improved a local dataset or reproduced a paper result.

## Graph Boundary

Each plugin is inserted immediately before native Detect and declares:

- input strides: `8, 16, 32`
- output strides: `8, 16, 32`
- input channels: audited from the installed YOLO26 Detect branches
- output channels: unchanged
- input size: fixed `imgsz=640`

The native YOLO26 one-to-one and one-to-many heads remain in place. The
NMS-free inference path, `reg_max=1`, and DFL-free loss are unchanged. These
plugins do not copy another detector's backbone, head, assigner, or training
recipe.

Gold-YOLO and RTMDet entries are explicit YOLO26 adaptations. Their paper
claims remain `paper_prior`; neither adapter claims an exact paper
reproduction.

## Runtime Gate

Before training, the runtime plugin:

1. validates the installed YOLO26 graph;
2. verifies P3/P4/P5 channels and strides;
3. wraps the already-loaded native Detect module;
4. records matched, missing, unexpected, and remapped checkpoint keys;
5. runs a native export dry-run;
6. checks latency, VRAM estimate, parameter count, and model size limits;
7. fails closed if any hard guard is exceeded.

The runtime writes `<component_id>_manifest.json`. Actual pilot promotion still
requires matched control evidence and paired device measurements. A runtime
manifest is not promotion evidence by itself.

## Optional Operators

No current neck plugin requires deformable convolution. If a future recipe
requests one through `deformable_module`, the dependency gate checks only the
local environment. A missing operator creates a typed
`implementation_request`; the system does not install packages, substitute a
standard convolution, or enqueue training.

## Recipes

Each neck is an independent `AtomicRecipe` with one changed variable:
`model_graph.neck_plugin`. Combining a neck with P2, a different neck, or a
head/loss change requires a separate coupled recipe and internal ablation.
