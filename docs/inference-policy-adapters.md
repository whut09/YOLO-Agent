# Isolated Inference Policy Adapters

> All Windows command examples are single-line PowerShell commands. PowerShell does not support the Bash `\` line continuation.

Inference-only paper methods are evaluated outside the training recipe and outside
the standard 640 metric namespace. They reuse a frozen detector checkpoint; they do
not create training nodes, consume ASHA training budget, or receive model-component
attribution.

This document places inference/postprocess actions inside the **Unified Action
Space**: which action families exist, how their metrics are namespaced, how the
decision loop treats them, and what deployment tradeoffs they carry. SAHI has a
dedicated certification contract documented in
[SAHI Independent Inference Certification](sahi-inference-certification.md).

## Unified Action Space boundary

The action space (`agents/action_space_schemas.py:23-59`) defines 27 families.
Four of them are inference-only:

```text
INFERENCE_ONLY_FAMILIES = {postprocess, threshold, inference, calibration}
```

(`action_space_schemas.py:64-71`.) Pydantic validators enforce the phase boundary
structurally: an inference-only family must declare `runtime_phase: inference | eval`,
and a train-graph family is forbidden from declaring the inference phase
(`action_space_schemas.py:202-218`). `input_resolution` is the only dual-phase
family (`DUAL_PHASE_FAMILIES`, `:102`): train-time `imgsz` is pinned to 640, and
only inference-side resolution sweeps are free.

Everything else is a training-side family. In user terms:

| User concept | Real action families | Side |
| --- | --- | --- |
| network / model graph | `backbone`, `neck`, `feature_fusion`, `head`, `model_scale` | training |
| loss | `bbox_loss`, `classification_loss`, `auxiliary_loss` | training |
| assigner | `assignment` | training |
| sampling | `sampling`, `data_selection`, `active_learning` | training (data) |
| augmentation | `augmentation`, `preprocessing` | training (data) |
| training strategy | `training_strategy`, `optimizer`, `regularization` | training |
| threshold / NMS / TTA / SAHI / tile / calibration / fusion | `postprocess`, `threshold`, `inference`, `calibration` | **inference-only** |

There is no family literally named `network`; the graph side is the four
model-graph families plus `model_scale`.

### User terms to real inference identifiers

| Concept | Real policy kind (`components/adapters/inference/policy.py:15-22`) | Component id | Metric namespace |
| --- | --- | --- | --- |
| SAHI slicing | `sahi_slicing` | `inference.sahi_slicing` | `sliced_inference` → `sliced_*` |
| Tiled multi-scale | `tiled_multi_scale` | `inference.tiled_multi_scale` | `tiled_multi_scale_inference` → `tiled_multi_scale_*` |
| Test-time augmentation | `test_time_augmentation` | `inference.test_time_augmentation` | `tta_inference` → `tta_*` |
| Confidence calibration | `confidence_calibration` | `inference.confidence_calibration` | `calibrated_inference` → `calibrated_*` |
| Class-aware thresholding | `class_aware_thresholding` | `inference.class_aware_thresholding` | `class_threshold_inference` → `class_threshold_*` |
| Cross-view merge / NMS / fusion | `merge_policy` (NMM, weighted box fusion) | `inference.merge_policy` | `merged_inference` → `merged_*` |

NMS changes are carried by `merge_policy` + `merge_iou_threshold`, not by mutating
Ultralytics' internal NMS. The loop-side postprocess strategy catalog
(`configs/postprocess_strategies.yaml`: `standard_nms`, `class_aware_nms`,
`soft_nms`, ...) feeds proposals only.

### Inference actions never change the training model claim

This is enforced in several independent places:

- The adapter base class freezes `modified_model_fields = modified_training_fields = frozenset()`
  (`components/adapters/inference/plugin.py:109-110`) and requires `imgsz=640` (`:137-140`).
- Component contracts declare `inference_only: true`, `training_only: false`,
  `changes_model_graph: false` (`configs/components/inference/isolated_policies.yaml`).
- `training_attribution_allowed` is `Literal[False]` on the policy config (`policy.py:103`).
- Metric deltas with policy prefixes are excluded from component contribution
  attribution (`agents/component_contribution.py:309-312`).
- Paper-side schemas force `standard_640_namespace_preserved=True`
  (`research/paper_inference_side_schemas.py:59-75`).

A sliced or TTA gain is a deployment gain. It can never be reported as a training
improvement, never promote an ASHA training candidate, and never change what the
trained checkpoint is claimed to be.

## Metric namespaces

`MetricNamespace` is a closed Literal (`agents/pareto.py:10-18`):
`standard_640`, `sliced_inference`, `tiled_multi_scale_inference`,
`tta_inference`, `calibrated_inference`, `class_threshold_inference`,
`merged_inference`. Every policy metric is written with its namespace prefix
(`policy.py:142-154`, rule `prefix = namespace.removesuffix("_inference")`):

```text
{prefix}_map50_95, {prefix}_ap_small, {prefix}_recall,
{prefix}_latency_ms, {prefix}_throughput, {prefix}_peak_vram_mb
```

Real examples: `sliced_map50_95`, `sliced_ap_small`, `sliced_latency_ms`,
`sliced_throughput` (`components/adapters/inference/slicing.py:96-99`),
`tta_map50_95`, `calibrated_recall`. The standard baseline stays unprefixed under
`standard_640` (`map50_95`, `latency_ms`, `model_size_mb`, ...).

**Why an inference result can never overwrite the standard baseline.** The
isolation does not rely on a promotion gate remembering to check; the policy
metrics are never written under standard keys in the first place:

1. The report schema rejects any policy-prefixed key inside `standard_640_metrics`
   (`certification/inference_policy_schemas.py:43-44`, `_is_policy_metric` `:76-80`).
2. The `--standard-metrics` input is validated the same way
   (`certification/inference_policy_runner.py:214-223`); SAHI mirrors this
   (`certification/sahi_schemas.py:53-54`, `sahi_runner.py:166-167`).
3. The policy artifact writer refuses to emit standard metric keys
   (`components/adapters/inference/policy_artifacts.py:57-72`).
4. Pareto selection is partitioned per namespace and never compares changed
   protocols with `standard_640` or with each other (`pareto.py:107-116`).
5. Component-contribution attribution skips policy-prefixed metrics
   (`component_contribution.py:198, 295`).
6. Paper-side deployment boundaries must record `standard_640_namespace_preserved=True`
   (`paper_inference_side_schemas.py:59-75`).

## Decision loop integration

The loop can propose inference-side moves from error facts, but their execution
class is fixed by the domain:

- Small-object false negatives: the real policy entry is `enable_tiling_inference`
  under `small_object_miss` (`configs/error_action_policies.yaml:19-25`,
  `target_variables: inference_tiling: sahi`) — a slicing/tiling candidate.
- False positives: `configs/postprocess_strategies.yaml` holds NMS strategies
  (`standard_nms`, `class_aware_nms`, `soft_nms`) that the loop's postprocess
  dimension can reference, and threshold/calibration candidates exist as
  `class_aware_thresholding` / `confidence_calibration` policies.
- The loop materializes these as `action_domain="postprocess"` proposals
  (`agents/error_driven_loop.py:367-381`).

Every such candidate is then forced to `recommendation_only`
(`agents/auto_optimization_loop.py:5043-5045`, via
`NON_TRAINING_DOMAINS = {"data", "label", "postprocess", "evidence"}` at `:911`;
any `command_type != "train"` is also rejected at `:5039-5041`).

**Inference candidates do not consume the ASHA training budget.** The chain:

1. `assess_candidate_execution` marks the proposal `recommendation_only`.
2. `_executable_nodes` (`auto_optimization_loop.py:7891-7913`) keeps only
   `executable` / `evidence_bootstrap` nodes in `round_plan.execution_nodes`.
3. ASHA registration runs only over executable nodes
   (`auto_optimization_loop.py:2253-2257`).
4. Rejected proposals survive only as `critic_results` and the round summary's
   `recommendation_only` list (`:7941-7943`, `:8263-8272`).

Two structural backstops close the loop: `register_trial` raises
`inference-only candidate cannot enter training ASHA` (`agents/asha_scheduler.py`),
and paper-level nodes containing `inference.` components are forced onto the
`inference_only_protocol` (`auto_optimization_loop.py:3039-3040`;
`research/paper_protocol_catalog.py:105-129`).

Real inference-policy evaluation therefore happens through the explicit
certification commands below, not inside the automatic loop.

## Deployment tradeoff and Pareto fronts

Each policy evaluation measures real deployment cost
(`components/adapters/inference/backend.py:38-95`):

- **accuracy**: `map50_95`, `ap_small`, `recall` via COCO eval
- **latency**: wall-clock per image, including view generation and post-processing
- **throughput**: images per second
- **VRAM**: `torch.cuda` peak memory allocation (0 on CPU)

Pareto fronts are partitioned by metric namespace and never merged
(`ParetoSelector.select_partitioned`, `pareto.py:107-116`). The standard front
dominates on accuracy/robustness/latency/model-size (`pareto.py:235-257`); the
paper-level inference fronts additionally use throughput (max) and peak VRAM
(min) as domination dimensions (`reports/pareto_report.py:149-163`). Reports are
written with separate "Standard 640 Pareto Front" and "Isolated Inference Policy
Pareto Fronts" sections in `paper_recipe_report.yaml/.md`
(`reports/paper_recipe_report.py:142-148`).

**TTA and SAHI trade accuracy for latency.** Multi-view and tiled execution run
the detector once per view or tile, so `tta_latency_ms` / `sliced_latency_ms` can
be a multiple of the standard single-pass cost, with corresponding throughput and
VRAM increases. That is precisely why these results live in their own namespace
and their own Pareto front: an accuracy gain that triples latency is not a
strictly better point, and the front makes the tradeoff visible instead of
letting it hide inside the standard metric.

One measurement caveat: policy latency timing has no warmup phase and covers the
whole batch loop (`backend.py:50, 87`), which is a different protocol from the
training-side fixed benchmark (`adapters/ultralytics/inference_latency.py`:
synthetic zero images, 3 warmups, 20 timed runs). Never compare those two
numbers directly.

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

Run the isolated evaluation explicitly (single line):

```powershell
yolo-agent advanced certify-inference-policy --workdir runs/certification/tta-yolo26n --model yolo26n.pt --images E:\dataset\coco\images\val2017 --annotations E:\dataset\coco\annotations\instances_val2017.json --config configs\my-tta-policy.yaml --standard-metrics runs\baseline\coco_eval.json --execute
```

Arguments (`cli.py:7347-7417`): `--workdir` (default
`runs/certification/inference-policy`), required `--model` / `--images` /
`--annotations` / `--config`, optional `--standard-metrics` JSON, and the opt-in
`--execute` switch — without it the command writes a `skipped` report and never
calls the model (`inference_policy_runner.py:62-74`).

`InferencePolicyConfig` fields (`policy.py:44-89`); only `policy_id` and `kind`
are required:

| Field | Default | Notes |
| --- | --- | --- |
| `standard_imgsz` | `640` | Literal — only 640 is accepted |
| `confidence_threshold` | `0.001` | |
| `temperature` | `1.0` | calibration requires a non-neutral temperature |
| `class_thresholds` | `{}` | required for `class_aware_thresholding` |
| `scales` | `[1.0]` | multi-scale views |
| `horizontal_flip` | `false` | TTA view |
| `tile_sizes` | `[640]` | each value must be ≥ 32 |
| `overlap_ratio` | `0.2` | tiling overlap |
| `merge_policy` | `none` | `none` / `nms` / `nmm` / `weighted_box_fusion` |
| `merge_iou_threshold` | `0.55` | |
| `one_to_one_head` | `true` | YOLO26 one-to-one head gets no extra NMS by default |
| `allow_cross_view_merge` | `false` | merge on the one-to-one head requires this explicitly |
| `max_detections` | `300` | |

A runtime guard (`InferencePolicyPlugin.prepare_command`,
`components/adapters/inference/plugin.py:47-97`) only accepts the
`certify-inference-policy ... --execute` command shape and only component ids
prefixed with `inference.`.

Outputs, under the workdir: `inference_policy_certification_report.yaml` plus
`artifacts/{prefix}_protocol.json`, `{prefix}_predictions.json`,
`{prefix}_metrics.json`, `{prefix}_resources.json`, and
`{prefix}_merge_statistics.json` (`policy_artifacts.py:24-54`).

**SAHI path note.** `sahi_slicing` is one of the six policy kinds, but running it
through `certify-inference-policy` uses the single-view 640 backend branch
(`backend.py:53-72`) — it is **not** real slicing. True SAHI slicing lives only
in `advanced certify-sahi` with its own config, report, optional-dependency
detection, and slicing backend. See
[SAHI Independent Inference Certification](sahi-inference-certification.md).

**Maturity note.** Inference-policy certification writes only workdir reports.
It never updates `runs/component_maturity_registry.yaml`, and
`configs/capability_maturity.yaml` has no inference/postprocess entries. The six
`inference.*` components declare `maturity: adapter_implemented` in their
contracts; component maturity progression is the separate `certify-component`
path ([GPU Certification](gpu-certification.md)).

## YOLO26 boundaries

- The standard comparison input remains `imgsz=640`; TTA and tiled views publish
  changed-protocol metrics only.
- A one-to-one YOLO26 head receives no extra NMS by default.
- Cross-view NMS requires `allow_cross_view_merge: true` and is recorded as an
  inference policy change.
- NMM and weighted box fusion merge only predictions from multiple views or tiles.
- Paper claims remain priors. Only local certification artifacts enter a Pareto
  front.

## Related documents

- [System Architecture](architecture.md) — the inference plane's place in the overall system.
- [Automatic Optimization Architecture](automatic-optimization-architecture.md) — the full decision loop and why inference-phase actions are budget-free.
- [Evidence Architecture](evidence.md) — which claims inference-policy metrics are allowed to support.
- [SAHI Independent Inference Certification](sahi-inference-certification.md) — the dedicated slicing certification contract.
- [GPU Certification](gpu-certification.md) — the four certification layers and component maturity.
