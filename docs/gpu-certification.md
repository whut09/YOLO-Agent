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

Run the explicit advanced command:

```powershell
yolo-agent advanced certify-gpu `
  --workdir runs/certification/mini-gpu `
  --model yolo26n.pt `
  --device 0 `
  --execute-real-gpu
```

The suite creates a deterministic, tiny COCO-compatible dataset and validates:

```text
train entrypoint -> debug -> matched pilot_3 cohort -> fixed post-eval
-> error facts -> verified paired delta -> ASHA decision -> matched pilot_10
```

All training and evaluation use `imgsz=640`. The result is written to:

```text
runs/certification/mini-gpu/certification_report.yaml
```

The mini suite certifies that the pipeline is executable. It does not prove a
`+0.02 mAP50-95` improvement on COCO and does not authorize a full COCO run.

## Pytest Gate

The real GPU test is marked `real_gpu` and is skipped by default. Run it explicitly:

```powershell
pytest -m real_gpu --run-real-gpu -q
```

Environment overrides:

```powershell
$env:YOLO_AGENT_CERT_MODEL="yolo26n.pt"
$env:YOLO_AGENT_CERT_DEVICE="0"
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

This gate prevents the documentation from presenting a partial implementation as a
locally reproduced capability.
