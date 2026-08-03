# Real GPU Certification

The GPU certification suite verifies that YOLO Agent can execute its evidence and
budget-control pipeline on real CUDA hardware. It is deliberately opt-in: normal
`pytest` and normal documentation generation never start training.

## Install

```powershell
pip install -e ".[train,certification]"
```

The certification suite requires a CUDA-capable PyTorch installation, Ultralytics,
and `pycocotools`. Model weights may be resolved by Ultralytics at run time and are
not committed to this repository.

## Mini COCO Acceptance

Before a component participates in the mini matched-pilot suite, certify its isolated
runtime path:

```powershell
yolo-agent advanced certify-component --component sampling.small_object --cpu
yolo-agent advanced certify-component --component sampling.small_object --gpu --device 0
```

The GPU command is real and opt-in. It requires a local checkpoint, generates a
small COCO-compatible fixture, and runs `train -> checkpoint -> resume` through
the typed Ultralytics runtime entrypoint at `imgsz=640`. It records hook calls,
backward/AMP evidence, GPU/VRAM, fixed-protocol latency, model size, adapter hash,
and immutable failure artifacts. It never downloads a checkpoint.

Run the high-value adapters in guarded priority order:

```powershell
yolo-agent advanced certify-paper-components `
  --model E:\path\yolo26n.pt `
  --teacher E:\path\yolo26s.pt `
  --device 0 `
  --execute-real-gpu
```

The suite stops at the first failure. A successful component advances only to
`gpu_certified`; `pilot_reproduced` still requires matched pilot evidence.

Default `pytest` skips real CUDA. The opt-in sampling acceptance test is:

```powershell
$env:YOLO_AGENT_RUN_REAL_GPU_COMPONENTS="1"
pytest -m real_gpu tests/test_component_gpu_real.py
```

The same command supports the certified training-only loss tracks:

```powershell
yolo-agent advanced certify-component --component loss.quality.correlation --cpu
yolo-agent advanced certify-component --component loss.calibration.bpc --cpu
yolo-agent advanced certify-component --component loss.quality.pseudo_iou --cpu
yolo-agent advanced certify-component --component distillation.yolo26_teacher_student --cpu
```

Their CPU reports prove runtime loss injection, backward, and zero-weight native
equivalence. Distillation additionally proves teacher no-grad and unchanged student
inference structure. These reports do not claim paper-exact reproduction or local
metric improvement; matched pilot certification is still required for either claim.

CPU certification is local and does not require CUDA. GPU certification is explicit
opt-in and is blocked until artifact-backed CPU `smoke_passed` exists for the same
protocol. These commands do not run a matched pilot or claim reproduction; they only
establish `smoke_passed` and `gpu_certified` runtime maturity.

Model-graph and assignment tracks use the same isolated command:

```powershell
yolo-agent advanced certify-component --component head.p2_small_object --cpu
yolo-agent advanced certify-component --component neck.multi_scale_fusion --cpu
yolo-agent advanced certify-component --component neck.gold_gather_distribute --cpu
yolo-agent advanced certify-component --component neck.rtmdet_large_kernel --cpu
yolo-agent advanced certify-component --component assigner.task_aligned --cpu
yolo-agent advanced certify-component --component assigner.optimal_transport --cpu
yolo-agent advanced certify-component --component assigner.dynamic_smooth_label --cpu
yolo-agent advanced certify-component --component assigner.dynamic_topk --cpu
yolo-agent advanced certify-component --component assigner.quality_aware --cpu
yolo-agent advanced certify-component --component assigner.dual_path --cpu
```

P2 and neck CPU reports require real graph forward, native loss, backward, AMP,
partial checkpoint accounting, export, and hard latency/VRAM/parameter/model-size
guards. Their recipes require matched controls. All assignment mechanisms remain
shadow-only at certification time: reports contain per-path baseline/candidate
positive ratios, conflict rate, matching stability, native-loss equivalence, and
native-path preservation.

Run `--gpu --device 0` only after the corresponding CPU overlay is valid. GPU smoke
does not create an active assignment pilot. That requires a passed same-protocol shadow
artifact, an explicit matched control, and ASHA plan materialization. A graph or
assignment `gpu_certified` state is runtime evidence, not a local accuracy claim.

For `sampling.small_object`, the mini suite is the golden path:

```text
ComponentValidationBridge
-> runtime_integrated -> unit_tested -> isolated CPU golden fixture -> smoke_passed
-> opt-in component CUDA smoke -> gpu_certified
-> matched baseline pilot_3 -> sampling pilot_3
-> COCO post-eval -> AP_small / target recall / FN paired delta
-> paired bootstrap -> diagnosis-bound promotion -> ASHA
-> matched baseline pilot_10 -> sampling pilot_10
```

The CPU golden fixture invokes the installed Ultralytics trainer dataloader bridge.
It verifies a protocol-bound `sampler_manifest.json`, deterministic DDP position
sharding, sampler resume state, and that validation loaders never call the train-only
sampling hook. Patch preview or mock smoke evidence cannot satisfy this stage.

Run the explicit advanced command:

```powershell
yolo-agent advanced certify-gpu `
  --workdir runs/certification/mini-gpu `
  --model yolo26n.pt `
  --device 0 `
  --recipe small_object_sampling `
  --execute-real-gpu
```

The suite creates a deterministic, tiny COCO-compatible dataset and validates:

```text
catalog import -> frozen ResearchSnapshot -> diagnosis-linked paper prior
-> eligibility gate -> executable adapter -> train entrypoint -> debug
-> matched pilot_3 cohort -> fixed post-eval -> error facts
-> AP_small / target-class recall / FN paired deltas -> paired bootstrap
-> latency and model-size guards -> ASHA decision -> matched pilot_10
-> policy memory update
```

All training and evaluation use `imgsz=640`. The result is written to:

```text
runs/certification/mini-gpu/certification_report.yaml
```

The mini suite certifies that the pipeline is executable. It does not prove a
`+0.02 mAP50-95` improvement on COCO and does not authorize a full COCO run.
For `small_object_sampling`, promotion is diagnosis-bound: AP_small, target-class
recall, and false-negative count must improve while overall mAP, latency, and model
size stay inside the report's guards. A failed report is retained as hashed evidence,
emits no `small_object_sampling_runtime` reproduction claim, and does not advance
component maturity. A passed mini report can advance `smoke_passed` to
`gpu_certified` only when all earlier artifact contracts are already present; it
cannot skip missing maturity states.

The stricter paper-driven acceptance suite freezes real MethodProfile and maturity
artifacts, disables scalar HPO, and is the only certification path that can advance
`sampling.small_object` from `gpu_certified` to `pilot_reproduced`. See
[Paper Auto-Optimization Acceptance](paper-auto-optimization-certification.md).

A general GPU certification report from another recipe does not authorize
`sampling.small_object`. Automatic ASHA registration requires a matching passed report
whose executed recipe is `small_object_sampling`, whose code hash is current, and whose
capability claims include `small_object_sampling_runtime`. Until then the candidate is
blocked before receiving a queue assignment.

## Pytest Gate

The real GPU test is marked `real_gpu` and is skipped by default. Run it explicitly:

```powershell
pytest -m real_gpu --run-real-gpu -q
```

Environment overrides:

```powershell
$env:YOLO_AGENT_CERT_MODEL="yolo26n.pt"
$env:YOLO_AGENT_CERT_DEVICE="0"
$env:YOLO_AGENT_CERT_RECIPE="small_object_sampling"
```

`YOLO_AGENT_RUN_REAL_GPU=1` is an alternative opt-in for CI workers dedicated to
GPU acceptance. Do not set it in ordinary unit-test jobs.

## Full COCO Certification

Full certification remains a consented, budgeted operation. Before starting it,
freeze one objective, dataset manifest, code version, Ultralytics version, batch
policy, and evaluation protocol. A protocol change invalidates prior consent.

Use this protocol for every baseline and candidate observation:

- COCO dataset and split manifests match exactly.
- `imgsz=640` and the same batch policy are fixed.
- Baseline seeds 1, 2, and 3 complete training, prediction export, COCO post-eval,
  error-fact import, latency measurement, and model-size measurement.
- Candidate seeds 1, 2, and 3 use the same protocol and each has a matched baseline.
- Image-level paired bootstrap and cross-seed confidence intervals are generated.
- The objective uses `mAP50-95` absolute delta, normally `+0.02`, and declares
  latency and model-size regression guards before training.
- A failed seed or incomplete artifact contract is preserved as evidence and blocks
  promotion. It is never silently discarded.

The final `certification_report.yaml` must use level `full_coco_multi_seed`, contain
three distinct baseline seeds and three distinct candidate seeds, include a passed
objective, and carry capability-specific claims. Full training still requires the
existing explicit `--confirm-full-run` consent path.

## Capability Promotion

The capability matrix separates code presence, automatic execution, and local
reproduction. A manifest entry cannot claim `locally_pilot_reproduced` or
`confirmed_multi_seed` by editing YAML alone:

- `locally_pilot_reproduced` requires a valid, passed mini or full certification
  report containing a matching capability claim.
- `confirmed_multi_seed` requires a valid `full_coco_multi_seed` report with a
  passed objective and at least three baseline and candidate seeds.
- The certification report is content-hashed. A modified payload fails validation.
- `pilot_reproduced` still requires a verified paired pilot result; GPU execution
  alone is not a reproduction claim.

This gate prevents the documentation from presenting a partial implementation as a
locally reproduced capability.

SAHI uses a separate inference-only certification path. It does not train a model or
promote a training component. See [SAHI Independent Inference Certification](sahi-inference-certification.md).
