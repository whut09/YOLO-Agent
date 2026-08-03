# Paper Auto-Optimization Acceptance

This opt-in suite certifies four paper mechanism families in one bounded ASHA study:

```text
fresh ResearchSnapshot -> diagnosis -> PaperMethodProfile
-> four gpu_certified runtime adapters
-> matched pilot_3 cohort -> fixed COCO post-eval -> complete error facts
-> paired bootstrap -> ASHA -> matched pilot_10 survivors
-> PolicyMemory -> pilot_reproduced
```

The cohort contains:

| Family | Component | Primary local evidence |
| --- | --- | --- |
| sampling | `sampling.small_object` | AP_small, recall, false negatives |
| auxiliary loss | `loss.quality.correlation` | mAP50-95, localization errors |
| distillation | `distillation.yolo26_teacher_student` | mAP50-95, recall, false negatives |
| model graph | `head.p2_small_object` | AP_small, recall, false negatives |

The suite fixes `imgsz=640`, disables scalar HPO, and never treats paper claims as
local observations. An adapter can serve multiple papers sharing the same canonical
mechanism; this is component adaptation, not exact paper reproduction.

## Prerequisites

Install the training and certification dependencies, then GPU-certify all four exact
runtime identities:

```powershell
python -m pip install -e ".[train,certification]"
yolo-agent advanced certify-component --component sampling.small_object --gpu --model E:\path\yolo26n.pt --device 0
yolo-agent advanced certify-component --component loss.quality.correlation --gpu --model E:\path\yolo26n.pt --device 0
yolo-agent advanced certify-component --component distillation.yolo26_teacher_student --gpu --model E:\path\yolo26n.pt --device 0
yolo-agent advanced certify-component --component head.p2_small_object --gpu --model E:\path\yolo26n.pt --device 0
```

The maturity registry must contain valid `gpu_certified` overlays for the current
adapter hashes and installed Ultralytics version. Distillation also requires the
configured teacher checkpoint. The Awesome-object-detection source must be a local
checkout or exported `papers.json`; acceptance does not access the network.

## Run

```powershell
yolo-agent advanced certify-paper-auto `
  --workdir runs/certification/paper-auto `
  --research-root research `
  --source E:\path\Awesome-object-detection `
  --registry runs/component_maturity_registry.yaml `
  --policy-root runs `
  --model E:\path\yolo26n.pt `
  --device 0 `
  --execute-real-gpu
```

Without `--execute-real-gpu`, the command writes a skipped report and starts no
training. The terminal identifies every paper component, family, adapter hash,
maturity, matched control, primary paired delta, target error delta, and rejection
reason. The full machine-readable report is:

```text
runs/certification/paper-auto/paper_auto_optimization_report.yaml
```

## Guards

- Candidate and control must match dataset manifest, subset, seed, epochs, batch
  policy, Ultralytics version, evaluation protocol, and fixed image size.
- Missing predictions, COCO evaluation, error facts, or bootstrap evidence returns
  `evidence_recovery` immediately. No later candidate is trained.
- Promotion uses each recipe's declared target error facts. Overall mAP, latency,
  and model size remain hard guards.
- ASHA is the only pilot budget authority. Non-survivors and failed pilot_10 results
  are both written to PolicyMemory.
- Only a verified, passed pilot_10 can advance the exact component identity to
  `pilot_reproduced`.
- The suite never requests candidate-full or seeds 2/3. Those remain behind an
  explicit `--confirm-full-run` in the training workflow.

A passed result is local pilot evidence, not full reproduction, multi-seed
confirmation, or a guaranteed metric gain.

## Tests

Default tests use a mock GPU backend and do not start CUDA training:

```powershell
python -m pytest tests/test_paper_auto_optimization_acceptance.py -q
```

Real GPU execution occurs only through the explicit command above.
