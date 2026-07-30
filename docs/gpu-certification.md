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
yolo-agent advanced certify-component --component small_object_sampling --cpu
yolo-agent advanced certify-component --component small_object_sampling --gpu --device 0
```

CPU certification is local and does not require CUDA. GPU certification is explicit
opt-in and is blocked until artifact-backed CPU `smoke_passed` exists for the same
protocol. These commands do not run a matched pilot or claim reproduction; they only
establish `smoke_passed` and `gpu_certified` runtime maturity.

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
